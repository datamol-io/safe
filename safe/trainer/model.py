from typing import Optional
from typing import Union
from typing import Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from transformers import GPT2DoubleHeadsModel
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
from transformers.models.gpt2.modeling_gpt2 import add_start_docstrings_to_model_forward
from transformers.models.gpt2.modeling_gpt2 import replace_return_docstrings
from transformers.models.gpt2.modeling_gpt2 import GPT2_INPUTS_DOCSTRING
from transformers.models.gpt2.modeling_gpt2 import _CONFIG_FOR_DOC


class SAFEDoubleHeadsModel(GPT2DoubleHeadsModel):
    """The safe model is a dual head GPT2 model with a language modeling head and an optional multi-task regression head"""

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, GPT2DoubleHeadsModelOutput]:
        r"""
        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
            1]`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
            `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
        mc_labels (`torch.LongTensor` of shape `(batch_size, n_tasks)`, *optional*):
            Labels for computing the supervized loss for regularization.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if mc_token_ids is None and self.config.pad_token_id is not None and input_ids is not None:
            mc_token_ids = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(
                lm_logits.device
            )

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        mc_loss = None
        if mc_labels is not None:
            loss_fct = MSELoss()
            mc_loss = loss_fct(
                mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1, mc_logits.size(-1))
            )

        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
