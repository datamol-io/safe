## Config File

The input config file for training a `SAFE` model is very similar to the GPT2 config file, with the addition of an optional `num_labels` attribute for training with descriptors regularization.

```json
{
  "activation_function": "gelu_new",
  "attn_pdrop": 0.1,
  "bos_token_id": 10000,
  "embd_pdrop": 0.1,
  "eos_token_id": 1,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": "tanh",
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_hidden_size": 128,
  "summary_use_proj": true,
  "transformers_version": "4.31.0",
  "use_cache": true,
  "vocab_size": 10000,
  "num_labels": 9
}
```


## SAFE Model
::: safe.trainer.model

---

## Trainer
::: safe.trainer.trainer_utils

---

## Data Collator
::: safe.trainer.collator

---

## Data Utils
::: safe.trainer.data_utils


