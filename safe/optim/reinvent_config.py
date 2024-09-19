import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import tyro
import json
from enum import Enum
from typing_extensions import Annotated
from transformers import is_wandb_available
from trl.core import flatten_dict


JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]


class REINVENTStrategy(Enum):
    """Strategies for REINVENT optimization."""

    DAP = "dap"
    MAULI = "mauli"
    MASCOF = "mascof"
    SDAP = "sdap"

    def is_dap(self):
        """Check if the strategy is DAP or SDAP."""
        return self in (REINVENTStrategy.DAP, REINVENTStrategy.SDAP)


@dataclass
class REINVENTConfig:
    """
    Configuration class for REINVENTTrainer.

    This class encapsulates all the configuration parameters required for training using the REINVENT algorithm.
    It is a standalone class and does not inherit from any other configuration classes.

    Args:
        # General Configuration
        exp_name: Name of the experiment (used for logging/tracking purposes).. Defaults to the name of the script being run.
        seed: Random seed for reproducibility. Default=0
        log_with: Logging backend to use. Supported options are: ["wandb", "tensorboard"]
        model_name: Name of the model (used for logging/tracking purposes).
        reward_model: Name of the reward model (used for logging/tracking purposes).
        remove_unused_columns: Whether to remove unused columns from the dataset. Default=True

        # Tracker and Accelerator Configuration
        tracker_project_name: Name of the project for tracking. Default to safe-reinvent
        tracker_kwargs: Additional keyword arguments for the tracker.
        accelerator_kwargs: Additional keyword arguments for the Accelerator.
        project_kwargs: additional information for the project configuration of the accelerator

        # Training Configuration
        steps: Number of training steps. Default to 10000
        learning_rate: Learning rate for the optimizer. Default=1e-5
        batch_size: Number of samples per optimization step. Default to 128
        mini_batch_size: Number of samples optimized in each mini-batch. Default to 128
        gradient_accumulation_steps: Number of gradient accumulation steps.
        max_grad_norm: Maximum gradient norm for clipping. Default to None for no grad clipping
        gradient_checkpointing: Whether to use gradient checkpointing. Default to False
        optimize_device_cache: Optimize device cache for slightly more memory-efficient training. Default to False

        # REINVENT-Specific Parameters
        sigma: Scaling factor for the score. Default to 10.0
        strategy: Strategy to use for optimization. One of ["dap", "sdap", "mauli", "mascof"]. Default to "dap"
        entropy_coeff:  Entropy regularization coefficient. Increasing the entropy regularization will change
            the loss to promote preserving diversity as much as possible, this can decrease performance however. Default to 0
        is_action_basis: Whether to compute loss on an action (token) basis. Default to False
        use_experience_replay: Whether to use experience replay during training. Default To False
        max_buffer_size: Maximum size of the experience replay buffer. Default to 10_000
        reinvent_epochs: Number of epochs per step (equivalent to PPO epochs). Default to 1
        score_clip: Value to clip the scores range into [-score_clip, +score_clip]. If `None`, no clipping is applied.
        use_score_scaling: Whether to scale the scores. Default to False
        use_score_norm: Whether to normalize the scores when scaling is used. Default to True

    Attributes:
        world_size: Number of processes to use for distributed training. Set by REINVENTTrainer.
        global_batch_size: Effective batch size across all processes. Set by REINVENTTrainer.
        is_encoder_decoder: Whether the model is an encoder-decoder model. Set by REINVENTTrainer.
        is_peft_model: Whether the model is a PEFT (Parameter-Efficient Fine-Tuning) model. Set by REINVENTTrainer.
    """

    # General Configuration
    exp_name: str = None  # Will default to script name if not provided
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    model_name: str = "gpt2"
    reward_model: Optional[str] = None
    remove_unused_columns: bool = True

    # Tracker and Accelerator Configuration
    tracker_project_name: str = "safe-reinvent"
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    accelerator_kwargs: Dict[str, Any] = field(default_factory=dict)
    project_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training Configuration
    steps: int = 10000
    learning_rate: float = 1e-3
    batch_size: int = 128
    mini_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    gradient_checkpointing: bool = False
    optimize_device_cache: bool = False

    # REINVENT-Specific Parameters
    sigma: float = 60.0
    strategy: Literal["dap", "sdap", "mauli", "mascof"] = "dap"
    entropy_coeff: Optional[float] = None
    is_action_basis: bool = False
    use_experience_replay: bool = False
    max_buffer_size: int = 10000
    reinvent_epochs: int = 1
    score_clip: Optional[float] = None
    use_score_scaling: bool = False
    use_score_norm: bool = True

    # Internal attributes set by the trainer
    world_size: Optional[int] = None
    global_batch_size: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    is_peft_model: Optional[bool] = None

    def __post_init__(self):
        # Default exp_name to script name if not provided
        if self.exp_name is None:
            import os
            import sys

            self.exp_name = os.path.basename(sys.argv[0])[: -len(".py")]

        if self.entropy_coeff is None:
            self.entropy_coeff = 0.0

        supported_strategies = [strategy.value for strategy in REINVENTStrategy]
        if self.strategy not in supported_strategies:
            raise ValueError(
                f"Strategy needs to be one of {supported_strategies}, got '{self.strategy}'"
            )

        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError("`batch_size` must be a multiple of `mini_batch_size`.")

        if self.use_score_scaling and self.score_clip is None:
            warnings.warn(
                "use_score_scaling is True but score_clip is None. Scores will not be clipped."
            )

        # Check if wandb is installed if logging with wandb
        if self.log_with == "wandb" and not is_wandb_available():
            raise ImportError(
                "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
            )

        self.tracker_kwargs.setdefault(self.log_with, {})["name"] = self.exp_name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a flattened dictionary.

        Returns:
            Dict[str, Any]: Flattened dictionary of configuration parameters.
        """
        return flatten_dict(self.__dict__)
