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

    Parameters:
        # General Configuration
        exp_name (str, optional):
            Name of the experiment. Defaults to the name of the script being run.
        seed (int, optional, default=0):
            Random seed for reproducibility.
        log_with (Optional[Literal["wandb", "tensorboard"]], optional, default=None):
            Logging backend to use. Supported options are:
                - "wandb"
                - "tensorboard"
        task_name (Optional[str], optional):
            Name of the task (used for logging/tracking purposes).
        model_name (str, optional, default="gpt2"):
            Name of the model (used for logging/tracking purposes).
        reward_model (Optional[str], optional):
            Name of the reward model (used for logging/tracking purposes).
        remove_unused_columns (bool, optional, default=True):
            Whether to remove unused columns from the dataset.

        # Tracker and Accelerator Configuration
        tracker_project_name (str, optional, default="trl"):
            Name of the project for tracking.
        tracker_kwargs (Dict[str, Any], optional, default_factory=dict):
            Additional keyword arguments for the tracker.
        accelerator_kwargs (Dict[str, Any], optional, default_factory=dict):
            Additional keyword arguments for the Accelerator.
        project_kwargs (Dict[str, Any], optional, default_factory=dict):

        # Training Configuration
        steps (int, optional, default=20000):
            Number of training steps.
        learning_rate (float, optional, default=1.41e-5):
            Learning rate for the optimizer.
        batch_size (int, optional, default=128):
            Number of samples per optimization step.
        mini_batch_size (int, optional, default=128):
            Number of samples optimized in each mini-batch.
        gradient_accumulation_steps (int, optional, default=1):
            Number of gradient accumulation steps.
        max_grad_norm (Optional[float], optional, default=None):
            Maximum gradient norm for clipping.
        gradient_checkpointing (bool, optional, default=False):
            Whether to use gradient checkpointing.
        optimize_device_cache (bool, optional, default=False):
            Optimize device cache for slightly more memory-efficient training.

        # REINVENT-Specific Parameters
        sigma (float, optional, default=60.0):
            Scaling factor for the score.
        strategy (Literal["dap", "sdap", "mauli", "mascof"], optional, default="dap"):
            Strategy to use for optimization.
        entropy_coeff (float, optional, default=0.0):
            Entropy regularization coefficient. Increasing the entropy regularization will change
            the loss to promote preserving diversity as much as possible, this can decrease performance however.
        is_action_basis (bool, optional, default=False):
            Whether to compute loss on an action (token) basis.
        use_experience_replay (bool, optional, default=False):
            Whether to use experience replay during training.
        max_buffer_size (int, optional, default=10000):
            Maximum size of the experience replay buffer.
        reinvent_epochs (int, optional, default=4):
            Number of epochs per step (equivalent to PPO epochs).
        score_clip (Optional[float], optional, default=None):
            Value to clip the scores. If `None`, no clipping is applied.
        use_score_scaling (bool, optional, default=False):
            Whether to scale the scores.
        use_score_norm (bool, optional, default=False):
            Whether to normalize the scores when scaling is used.

    Attributes:
        world_size (Optional[int]):
            Number of processes to use for distributed training. Set by REINVENTTrainer.
        global_batch_size (Optional[int]):
            Effective batch size across all processes. Set by REINVENTTrainer.
        is_encoder_decoder (Optional[bool]):
            Whether the model is an encoder-decoder model. Set by REINVENTTrainer.
        is_peft_model (Optional[bool]):
            Whether the model is a PEFT (Parameter-Efficient Fine-Tuning) model. Set by REINVENTTrainer.
    """

    # General Configuration
    exp_name: str = None  # Will default to script name if not provided
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    task_name: Optional[str] = None
    model_name: str = "gpt2"
    reward_model: Optional[str] = None
    remove_unused_columns: bool = True

    # Tracker and Accelerator Configuration
    tracker_project_name: str = "trl"
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    accelerator_kwargs: Dict[str, Any] = field(default_factory=dict)
    project_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training Configuration
    steps: int = 20000
    learning_rate: float = 1.41e-5
    batch_size: int = 128
    mini_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    gradient_checkpointing: bool = False
    optimize_device_cache: bool = False

    # REINVENT-Specific Parameters
    sigma: float = 60.0
    strategy: Literal["dap", "sdap", "mauli", "mascof"] = "dap"
    entropy_coeff: float = 0.0
    is_action_basis: bool = False
    use_experience_replay: bool = False
    max_buffer_size: int = 10000
    reinvent_epochs: int = 4
    score_clip: Optional[float] = None
    use_score_scaling: bool = False
    use_score_norm: bool = False

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a flattened dictionary.

        Returns:
            Dict[str, Any]: Flattened dictionary of configuration parameters.
        """
        return flatten_dict(self.__dict__)
