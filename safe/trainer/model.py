# import os
# import sys
# from typing import List

# import fire
# import torch
# import transformers
# from datasets import load_dataset
# from transformers import DataCollatorForLanguageModeling
# from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
# from .arguments import DataArguments
# from .arguments import ModelArguments


# def train(
#     # model/data params
#     base_model: str = "",  # the only required argument
#     data_path: str = "./data/",
#     output_dir: str = "./checkpoint",
#     # training hyperparams
#     batch_size: int = 800,
#     micro_batch_size: int = 100,
#     num_epochs: int = 10,
#     learning_rate: float = 3e-4,
#     cutoff_len: int = 256,
#     # lora hyperparams
#     lora_r: int = 16,
#     lora_alpha: int = 16,
#     lora_dropout: float = 0.05,
#     lora_target_modules: List[str] = [
#         "q_proj",
#         "v_proj",
#     ],
#     # llm hyperparams
#     train_on_inputs: bool = True,  # if False, masks out inputs in loss
#     group_by_length: bool = False,  # faster, but produces an odd training loss curve
#     # wandb params
#     wandb_project: str = "",
#     wandb_run_name: str = "",
#     wandb_watch: str = "",  # options: false | gradients | all
#     wandb_log_model: str = "",  # options: false | true
#     resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
#     prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
#     warmup_steps=100,
#     logging_steps=10,
#     save_steps=200,
#     save_total_limit=3,
#     eval_steps=200,
# ):
#     gradient_accumulation_steps = batch_size // micro_batch_size

#     device_map = "auto"
#     world_size = int(os.environ.get("WORLD_SIZE", 1))
#     ddp = world_size != 1
#     if ddp:
#         device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
#         gradient_accumulation_steps = gradient_accumulation_steps // world_size

#     # Check if parameter passed or if set within environ
#     use_wandb = len(wandb_project) > 0 or (
#         "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
#     )
#     # Only overwrite environ if wandb param passed
#     if len(wandb_project) > 0:
#         os.environ["WANDB_PROJECT"] = wandb_project
#     if len(wandb_watch) > 0:
#         os.environ["WANDB_WATCH"] = wandb_watch
#     if len(wandb_log_model) > 0:
#         os.environ["WANDB_LOG_MODEL"] = wandb_log_model
#     model = LlamaForCausalLM.from_pretrained(
#         base_model,
#         load_in_8bit=True,
#         torch_dtype=torch.float16,
#         device_map=device_map,
#     )

#     tokenizer = LlamaTokenizer.from_pretrained(base_model)

#     tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
#     tokenizer.padding_side = "left"

#     def tokenize(prompt, add_eos_token=True):
#         # there's probably a way to do this with the tokenizer settings
#         # but again, gotta move fast
#         result = tokenizer(
#             prompt,
#             truncation=True,
#             max_length=cutoff_len,
#             padding=False,
#             return_tensors=None,
#         )
#         if (
#             result["input_ids"][-1] != tokenizer.eos_token_id
#             and len(result["input_ids"]) < cutoff_len
#             and add_eos_token
#         ):
#             result["input_ids"].append(tokenizer.eos_token_id)
#             result["attention_mask"].append(1)

#         result["labels"] = result["input_ids"].copy()

#         return result

#     model = prepare_model_for_int8_training(model)

#     config = LoraConfig(
#         r=lora_r,
#         lora_alpha=lora_alpha,
#         target_modules=lora_target_modules,
#         lora_dropout=lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(model, config)

#     if resume_from_checkpoint:
#         # Check the available weights and load them
#         checkpoint_name = os.path.join(
#             resume_from_checkpoint, "pytorch_model.bin"
#         )  # Full checkpoint
#         if not os.path.exists(checkpoint_name):
#             checkpoint_name = os.path.join(
#                 resume_from_checkpoint, "adapter_model.bin"
#             )  # only LoRA model - LoRA config above has to fit
#             resume_from_checkpoint = False  # So the trainer won't try loading its state
#         # The two files above have a different name depending on how they were saved, but are actually the same.
#         if os.path.exists(checkpoint_name):
#             print(f"Restarting from {checkpoint_name}")
#             adapters_weights = torch.load(checkpoint_name)
#             set_peft_model_state_dict(model, adapters_weights)
#         else:
#             print(f"Checkpoint {checkpoint_name} not found")

#     model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
#     if not ddp and torch.cuda.device_count() > 1:
#         # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
#         model.is_parallelizable = True
#         model.model_parallel = True

#     trainer = transformers.Trainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         args=,
#         data_collator=transformers.DataCollatorForSeq2Seq(
#             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
#         ),
#     )
#     model.config.use_cache = False

#     old_state_dict = model.state_dict
#     model.state_dict = (
#         lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
#     ).__get__(model, type(model))

#     if torch.__version__ >= "2" and sys.platform != "win32":
#         model = torch.compile(model)
#     # with torch.autocast("cuda"):
#     trainer.train(resume_from_checkpoint=resume_from_checkpoint)

#     model.save_pretrained(output_dir)

#     print("\n If there's a warning about missing keys above, please disregard :)")


# if __name__ == "__main__":
#     fire.Fire(train)
