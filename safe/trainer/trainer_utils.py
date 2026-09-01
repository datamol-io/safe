from transformers import Trainer


class SAFETrainer(Trainer):
    """
    Custom trainer for training SAFE model.

    This custom trainer changes the loss function to support the property head

    """

    def __init__(self, *args, prop_loss_coeff: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop_loss_coeff = prop_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        del num_items_in_batch
        labels = (
            inputs.pop("labels") if self.label_smoother is not None and "labels" in inputs else None
        )
        outputs = model(**inputs)
        # Preserve the past state when the Trainer is configured to retain it.
        past_index = getattr(self.args, "past_index", -1)
        if past_index >= 0:
            self._past = outputs[past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
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
