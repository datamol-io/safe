from trl import PreTrainedModelWrapper


class AutoModelForCausalLM(PreTrainedModelWrapper):

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            return_past_key_values (bool): A flag indicating if the computed hidden-states should be returned.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """

        kwargs["output_hidden_states"] = (
            True  # this had already been set in the LORA / PEFT examples
        )
        kwargs["past_key_values"] = past_key_values

        if (
            self.is_peft_model
            and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING"
        ):
            kwargs.pop("past_key_values")
        return self.pretrained_model.forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if (
                name == "pretrained_model"
            ):  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.pretrained_model, name)
