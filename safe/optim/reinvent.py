import inspect
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import heapq
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from trl.core import (
    PPODecorators,
    entropy_from_logits,
    masked_mean,
    set_seed,
)
from trl.import_utils import is_torch_greater_2_0
from trl.models import (
    PreTrainedModelWrapper,
    create_reference_model,
    unwrap_model_for_generation,
)
from trl.trainer import BaseTrainer, PPOConfig, RunningMoments

from .reinvent_config import REINVENTStrategy, REINVENTConfig

if is_deepspeed_available():
    import deepspeed


class REINVENTTrainer(BaseTrainer):
    """
    The REINVENTTrainer implements the REINVENT algorithm for optimizing language models with reinforcement learning.
    """

    def __init__(
        self,
        config: Optional[REINVENTConfig] = None,
        model: Optional[PreTrainedModelWrapper] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_data_collator: Optional[Callable] = None,
    ):
        """
        Initialize REINVENTTrainer.

        Args:
            config: Configuration object for REINVENTTrainer.
            model: Hugging Face transformer model with a value head.
            ref_model: Reference model (prior) to be used in REINVENT.
            tokenizer: Hugging Face tokenizer.
            dataset: PyTorch dataset or Hugging Face dataset.
            optimizer: Optimizer used for training.
            data_collator: Data collator function.
            num_shared_layers: Number of shared layers between the model and the reference model.
            lr_scheduler: Learning rate scheduler used for training.
            training_data_collator: Custom data collator used for training.
        """
        super().__init__(config)

        # Initial seed for reproducible experiments
        set_seed(config.seed)

        # Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        # Runtime variables filled by the accelerator
        config.world_size = self.accelerator.num_processes
        config.global_batch_size = config.batch_size * config.world_size

        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=(
                {"safe_reinvent_trainer_config": config.to_dict()}
                if not is_using_tensorboard
                else config.to_dict()
            ),
            init_kwargs=config.tracker_kwargs,
        )

        # Initialize reference (prior) model
        if isinstance(ref_model, PreTrainedModelWrapper):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the "
                    "model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported "
                f"architectures are: {PreTrainedModelWrapper} "
            )
        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (isinstance(dataset, (torch.utils.data.Dataset, Dataset))):
            raise ValueError("dataset must be a torch.utils.data.Dataset or datasets.Dataset")
        if dataset is None:
            warnings.warn(
                "No dataset is provided. Make sure to set config.batch_size to the correct value before training.",
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should"
                " prepare your dataloader yourself with `dataloader = reinvent_trainer.accelerator.prepare(dataloader)`"
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `REINVENTTrainer`. Please "
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Initialize optimizer and data collator
        if training_data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        else:
            self.data_collator = training_data_collator
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        # Initialize variables for early stopping and score scaling
        self.use_score_norm = config.use_score_norm
        self.reinvent_epochs = config.reinvent_epochs
        self.mini_batch_size = config.mini_batch_size

        # Ensure mini_batch_size divides batch_size
        if self.config.batch_size % self.mini_batch_size != 0:
            raise ValueError("`batch_size` must be a multiple of `mini_batch_size`.")

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        # Strategy and parameters specific to REINVENT
        self.sigma = config.sigma
        self.strategy = REINVENTStrategy(config.strategy)
        self.is_action_basis = config.is_action_basis

        # Initialize experience replay buffer (if needed)
        self.use_experience_replay = config.use_experience_replay
        self.experience_buffer = None
        if self.use_experience_replay:
            self.experience_buffer = []
            self.max_buffer_size = getattr(config, "max_buffer_size", 10000)  # Default buffer size

        # Safety checkers for DeepSpeed integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # For backward compatibility with older versions of transformers
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.pretrained_model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # Quantized models are already set on the correct device
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        self.is_distributed = self.accelerator.num_processes > 1

        # Initialize the current step
        self.current_step = 0

        # Device setup
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            if is_torch_xpu_available():
                self.current_device = torch.device("xpu:0")
            elif is_torch_npu_available():
                self.current_device = torch.device("npu:0")
            else:
                self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_device_cache = self.config.optimize_device_cache
        self.running = RunningMoments(self.accelerator)

    def prepare_dataloader(
        self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None
    ):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.

        Args:
            query_tensor: A tensor or list of tensors containing query tokens.
            length_sampler: Callable that returns the number of newly generated tokens.
            batch_size: Batch size used for generation, defaults to `4`.
            return_prompt: If set to `False` the prompt is not returned but only the newly generated tokens.
            generation_kwargs: Keyword arguments for generation.

        Returns:
            List[`torch.LongTensor`]: A list of tensors containing response tokens.
        """
        if isinstance(query_tensor, torch.Tensor):
            query_tensor = [query_tensor]

        responses = []
        batch_size = min(len(query_tensor), batch_size)

        for i in range(0, len(query_tensor), batch_size):
            batch_queries = query_tensor[i : i + batch_size]
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            input_ids = torch.nn.utils.rnn.pad_sequence(
                batch_queries, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).to(self.current_device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                outputs = unwrapped_model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs
                )

            for j, output in enumerate(outputs):
                response = output[len(batch_queries[j]) :] if not return_prompt else output
                responses.append(response)

        return responses

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[float],
    ):
        """
        Perform a REINVENT training step with multiple epochs and mini-batches.

        Args:
            queries: List of tensors containing the encoded queries.
            responses: List of tensors containing the generated responses.
            scores: List of scores (rewards) for the responses.

        Returns:
            A dictionary of training statistics.
        """

        # Ensure tensors are on the correct device
        queries = [q.to(self.current_device) for q in queries]
        responses = [r.to(self.current_device) for r in responses]
        scores = torch.tensor(scores, dtype=torch.float32).to(self.current_device)

        # Score scaling and clipping (same as before)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = {"dtype": scores.dtype, "device": scores.device}
            score_scaling_factor = (
                self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            )
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        # Optionally clip the scores
        if self.config.score_clip is not None:
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(
                dtype=scores_dtype
            )

        # Prepare inputs for the agent and prior
        model_inputs = self.prepare_model_inputs(queries, responses)

        # Pad inputs if distributed
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        # Compute log probabilities and entropies
        with torch.no_grad():
            prior_log_probs = self.compute_log_probs(self.ref_model, model_inputs)
            prior_log_probs = prior_log_probs.detach()

        # Prepare dataset for mini-batching
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(queries)),  # Indices
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
        )

        # Initialize variables for logging
        total_loss = 0.0

        for epoch in range(self.reinvent_epochs):
            for batch_indices in dataloader:
                idx = batch_indices[0]

                [queries[i] for i in idx]
                [responses[i] for i in idx]
                mb_scores = scores[idx]
                mb_model_inputs = {key: value[idx] for key, value in model_inputs.items()}
                mb_prior_log_probs = prior_log_probs[idx]

                agent_log_probs, agent_logits = self.compute_log_probs(
                    self.model, mb_model_inputs, return_logits=True
                )

                # Compute entropies for the mini-batch
                entropies = entropy_from_logits(agent_logits)

                # Adjust log_probs based on is_action_basis
                mb_attention_mask = mb_model_inputs["attention_mask"][:, 1:].float()  # Shifted mask
                if not self.is_action_basis:
                    agent_log_probs = agent_log_probs * mb_attention_mask
                    mb_prior_log_probs = mb_prior_log_probs * mb_attention_mask
                    entropies = entropies * mb_attention_mask

                # Compute loss based on the selected strategy
                loss = self.loss(
                    agent_log_probs,
                    mb_prior_log_probs,
                    mb_scores,
                    entropies,
                    mb_attention_mask,
                )

                # Backpropagation (rest of your code remains the same)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                if self.config.max_grad_norm is not None and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                total_loss += loss.detach().cpu().item()

        # Update experience buffer if enabled
        if self.use_experience_replay:
            self.update_experience_buffer(queries, responses, scores)

        # Prepare batch dictionary for logging
        batch = {
            "query": [self.tokenizer.decode(q, skip_special_tokens=True) for q in queries],
            "response": [self.tokenizer.decode(r, skip_special_tokens=True) for r in responses],
        }

        # Log statistics
        stats = {
            "loss": total_loss / (self.reinvent_epochs * len(dataloader)),
            "mean_score": scores.mean().detach().cpu().item(),
            "mean_entropy": entropies.mean().detach().cpu().item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "batch_size": len(queries),
        }
        self.log_stats(stats, batch, scores)

        # Return statistics
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: Optional[List[str]] = None,
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
            columns_to_log (Optional[List[str]], optional):
                Columns to log from the batch. Defaults to ["query", "response"].
        """
        if columns_to_log is None:
            columns_to_log = ["query", "response"]

        # Gather rewards across processes
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device)
        rewards = self.accelerator.gather(rewards).flatten()

        # Prepare batch data for logging
        if self.config.log_with == "wandb":
            import wandb

            if any(column_to_log not in batch for column_to_log in columns_to_log):
                raise ValueError(
                    f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}."
                )

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            if "query" not in batch and "response" not in batch:
                # Warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and "
                    "'response'. "
                )
            elif self.config.log_with == "wandb":
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                logs.update(
                    {"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)}
                )

            logs.update(stats)

            # Manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            if self.config.log_with == "tensorboard":
                # Update the current step
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=self.current_step if self.config.log_with == "tensorboard" else None,
            )

    def prepare_model_inputs(self, queries: List[torch.Tensor], responses: List[torch.Tensor]):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
            ).to(self.current_device)

        input_data.pop("labels", None)  # We don't want to compute LM losses
        return input_data

    def compute_log_probs(self, model, inputs, return_logits=False):
        outputs = model(**inputs)
        logits = outputs.logits

        if self.is_encoder_decoder:
            input_ids = inputs["decoder_input_ids"]
            attention_mask = inputs["decoder_attention_mask"]
        else:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

        # Shift logits and input_ids for log-likelihood calculation
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_input_ids = input_ids[..., 1:].contiguous()
        attention_mask = attention_mask[..., 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        log_probs = log_probs.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

        if return_logits:
            return log_probs, shifted_logits
        return log_probs

    def loss(self, agent_log_probs, prior_log_probs, scores, entropies, attention_mask):
        strategy = self.strategy

        if strategy.is_dap():
            if strategy == REINVENTStrategy.DAP:
                augmented_log_probs = prior_log_probs + self.sigma * scores.unsqueeze(1)
                loss = (augmented_log_probs - agent_log_probs).pow(2)

            # strategy == REINVENTStrategy.SDAP
            else:
                augmented_log_probs = prior_log_probs + self.sigma * scores.unsqueeze(1)
                reward = (augmented_log_probs - agent_log_probs).pow(2)
                loss = -reward * agent_log_probs

            # Include entropy regularization
            if self.config.entropy_coeff > 0:
                loss = loss - self.config.entropy_coeff * entropies

            # Apply masking and reduction
            loss = masked_mean(loss, attention_mask) if self.is_action_basis else loss.mean()

        else:
            if strategy == REINVENTStrategy.MASCOF:
                rewards_sum = scores.to(agent_log_probs.device)

            elif strategy == REINVENTStrategy.MAULI:
                rewards = prior_log_probs + self.sigma * scores.unsqueeze(1)
                rewards_sum = (
                    (rewards * attention_mask).sum(dim=1)
                    if self.is_action_basis
                    else rewards.sum(dim=1)
                )

            if self.is_action_basis:
                agent_log_probs_sum = (agent_log_probs * attention_mask).sum(dim=1)
                entropies_sum = (entropies * attention_mask).sum(dim=1)
            else:
                agent_log_probs_sum = agent_log_probs.sum(dim=1)
                entropies_sum = entropies.sum(dim=1)

            # Compute loss per sequence
            loss = -rewards_sum * agent_log_probs_sum

            if self.config.entropy_coeff > 0:
                loss = loss - self.config.entropy_coeff * entropies_sum
            loss = loss.mean()

        return loss

    def update_experience_buffer(self, queries, responses, scores):
        # Store experiences with inverted scores for max-heap behavior
        for q, r, s in zip(queries, responses, scores):
            experience = (-s.cpu().item(), q.cpu(), r.cpu())  # Negative score for max-heap behavior
            if len(self.experience_buffer) < self.max_buffer_size:
                heapq.heappush(self.experience_buffer, experience)
            else:
                # Push new experience and pop the smallest (lowest score) experience
                heapq.heappushpop(self.experience_buffer, experience)

    def sample_from_experience_buffer(self, batch_size: int):
        """Sample examples from the experience buffer

        Args:
            batch_size: Number of example to sample from the buffer
        """
        experiences = self.experience_buffer.copy()
        queries, responses, scores = zip(*[(q, r, -s) for s, q, r in experiences])
        indices = np.random.choice(len(queries), batch_size, replace=False)
        sampled_queries = [queries[i] for i in indices]
        sampled_responses = [responses[i] for i in indices]
        sampled_scores = [scores[i] for i in indices]
        return sampled_queries, sampled_responses, torch.tensor(sampled_scores)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None and hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9
                        * hidden_size
                        * hidden_size,
                    }
                )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
