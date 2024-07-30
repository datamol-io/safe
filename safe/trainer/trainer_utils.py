from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model


class SAFETrainer(Trainer):
    """
    Custom trainer for training SAFE model.

    This custom trainer changes the loss function to support the property head

    """

    def __init__(self, *args, prop_loss_coeff: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop_loss_coeff = prop_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        labels = (
            inputs.pop("labels") if self.label_smoother is not None and "labels" in inputs else None
        )
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        mc_loss = outputs.get("mc_loss", None) if isinstance(outputs, dict) else outputs[1]
        if mc_loss is not None:
            loss = loss + self.prop_loss_coeff * mc_loss
        return (loss, outputs) if return_outputs else loss
