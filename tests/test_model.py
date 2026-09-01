import torch
from transformers import GPT2Config, TrainingArguments

from safe.tokenizer import SAFETokenizer
from safe.trainer.cli import ModelArguments
from safe.trainer.model import PropertyHead, SAFEDoubleHeadsModel
from safe.trainer.trainer_utils import SAFETrainer


def tiny_config():
    return GPT2Config(
        vocab_size=16,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        num_labels=2,
        summary_hidden_size=4,
        summary_type="cls_index",
    )


def test_property_head_preserves_single_item_batch():
    head = PropertyHead(tiny_config())
    hidden_states = torch.randn(1, 3, 8)

    assert head(hidden_states, torch.tensor([1])).shape == (1, 2)
    assert head(hidden_states).shape == (1, 2)


def test_model_and_trainer_compute_current_loss_signature(tmp_path):
    model = SAFEDoubleHeadsModel(tiny_config())
    inputs = {
        "input_ids": torch.tensor([[1, 4, 5, 0], [1, 6, 7, 2]]),
        "labels": torch.tensor([[1, 4, 5, 0], [1, 6, 7, 2]]),
        "mc_labels": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }

    outputs = model(**inputs)
    assert outputs.loss is not None
    assert outputs.mc_loss is not None

    # Keep this unit test independent from whichever accelerator the host makes
    # available; the Trainer normally moves dataloader batches to that device.
    args = TrainingArguments(output_dir=tmp_path, report_to=[], use_cpu=True)
    trainer = SAFETrainer(model=model, args=args)
    loss = trainer.compute_loss(model, inputs.copy(), num_items_in_batch=2)
    assert loss.ndim == 0


def test_tokenizer_save_pretrained_round_trip(tmp_path):
    tokenizer = SAFETokenizer(tokenizer_type="wordlevel")
    tokenizer.train_from_iterator(["C%(100).N%99"])

    saved_files = tokenizer.save_pretrained(tmp_path)
    restored = SAFETokenizer.load(saved_files[0])

    assert restored.encode("C%(100).N%99") == tokenizer.encode("C%(100).N%99")


def test_cli_tokenizer_default_is_not_a_tuple():
    assert ModelArguments().tokenizer is None
