import pickle
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import GenerationConfig, GPT2Config, TrainingArguments

from safe.tokenizer import SAFETokenizer
from safe.trainer.cli import ModelArguments
from safe.trainer.model import PropertyHead, SAFEDoubleHeadsModel
from safe.trainer.trainer_utils import SAFETrainer
from safe.sample import SAFEDesign
from safe._pattern import PatternConstraint, PatternSampler

pytestmark = pytest.mark.integration


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


def test_tokenizer_pickle_preserves_safe_splitter():
    tokenizer = SAFETokenizer(tokenizer_type="wordlevel")
    tokenizer.train_from_iterator(["C%(100).N%99"])

    restored = pickle.loads(pickle.dumps(tokenizer))

    assert restored.encode("C%(100).N%99") == tokenizer.encode("C%(100).N%99")


def test_tokenizer_decodes_empty_input():
    tokenizer = SAFETokenizer(tokenizer_type="wordlevel")

    assert tokenizer.decode([]) == ""


def test_tokenizer_decodes_sequences_containing_only_stop_tokens():
    tokenizer = SAFETokenizer(tokenizer_type="wordlevel")

    assert tokenizer.decode([tokenizer.eos_token_id] * 3) == ""


def test_pattern_randomization_is_local_and_reproducible():
    first = PatternConstraint.randomize("c1cc([*])ccc1[*]", n=5, seed=7)
    second = PatternConstraint.randomize("c1cc([*])ccc1[*]", n=5, seed=7)

    assert first == second


def test_pattern_loss_uses_input_device_and_dtype():
    sampler = PatternSampler.__new__(PatternSampler)
    inputs = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)

    loss = sampler.nll_loss(inputs, torch.tensor([0, 1]))

    assert loss.dtype == inputs.dtype
    assert loss.device == inputs.device
    assert torch.equal(loss, torch.tensor([0.1, 0.4], dtype=torch.float64))


def test_pattern_constraint_rejects_an_empty_vocabulary_mask():
    constraint = PatternConstraint.__new__(PatternConstraint)
    constraint.temperature = 1.0
    constraint.force_constraint_sample = True

    with pytest.raises(ValueError, match="masks every token"):
        constraint._logprobs_to_probs(
            torch.tensor([0.1, 0.2]),
            mask=torch.tensor([False, False]),
        )


def test_cli_tokenizer_default_is_not_a_tuple():
    assert ModelArguments().tokenizer is None
    assert ModelArguments().wandb_project is None


def test_design_loads_string_generation_config(monkeypatch):
    model = MagicMock(spec=SAFEDoubleHeadsModel)
    tokenizer = MagicMock(spec=SAFETokenizer)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    generation_config = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0)
    loader = MagicMock(return_value=generation_config)
    monkeypatch.setattr(GenerationConfig, "from_pretrained", loader)

    SAFEDesign(model, tokenizer, generation_config="local-generation-config")

    loader.assert_called_once_with("local-generation-config")


def test_random_generation_does_not_set_beam_only_early_stopping():
    pretrained_tokenizer = MagicMock()
    pretrained_tokenizer.model_max_length = 32
    pretrained_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 4, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    pretrained_tokenizer.decode.return_value = "CC"
    pretrained_tokenizer.batch_decode.return_value = ["CC"]

    designer = SAFEDesign.__new__(SAFEDesign)
    designer.tokenizer = MagicMock()
    designer.tokenizer.get_pretrained.return_value = pretrained_tokenizer
    designer.model = MagicMock()
    designer.model.device = torch.device("cpu")
    designer.model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[1, 4, 2]]))
    designer.generation_config = None

    assert designer._generate(n_samples=1, safe_prefix="C", how="random") == ["CC"]
    assert "early_stopping" not in designer.model.generate.call_args.kwargs
    assert "output_scores" not in designer.model.generate.call_args.kwargs
    assert designer.model.generate.call_args.kwargs["max_new_tokens"] == 100

    designer.model.generate.reset_mock()
    designer._generate(n_samples=2, safe_prefix="C", how="beam")
    assert designer.model.generate.call_args.kwargs["early_stopping"] is True

    designer.model.generate.reset_mock()
    with pytest.warns(FutureWarning, match="max_length"):
        designer._generate(n_samples=1, safe_prefix="C", max_length=12)
    assert designer.model.generate.call_args.kwargs["max_length"] == 12
    assert "max_new_tokens" not in designer.model.generate.call_args.kwargs


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_beam_groups": 2}, "Diverse beam search"),
        ({"penalty_alpha": 0.6, "top_k": 4}, "Contrastive search"),
        ({"how": "unsupported"}, "how must be one of"),
    ],
)
def test_generate_rejects_unmaintained_sampling_modes(kwargs, match):
    designer = SAFEDesign.__new__(SAFEDesign)
    designer.tokenizer = MagicMock()
    designer.tokenizer.get_pretrained.return_value.model_max_length = 32

    with pytest.raises(ValueError, match=match):
        designer._generate(**kwargs)


def test_motif_extension_is_a_deprecated_scaffold_decoration_alias():
    designer = SAFEDesign.__new__(SAFEDesign)
    designer.scaffold_decoration = MagicMock(return_value=["CC"])

    with pytest.warns(FutureWarning, match="scaffold_decoration"):
        result = designer.motif_extension("C[*]", n_samples_per_trial=2)

    assert result == ["CC"]
    designer.scaffold_decoration.assert_called_once_with(
        "C[*]",
        n_samples_per_trial=2,
        n_trials=1,
        sanitize=False,
        do_not_fragment_further=True,
        random_seed=None,
        add_dot=True,
        try_hard=False,
    )


def test_try_hard_sampling_budget_and_stable_deduplication():
    assert SAFEDesign._candidate_count(4, try_hard=False) == 4
    assert SAFEDesign._candidate_count(4, try_hard=True) == 12
    samples = ["CC", None, "CN", "CC", "CO"]

    assert SAFEDesign._finalize_samples(samples, limit=2, try_hard=True) == ["CC", "CN"]
    assert SAFEDesign._finalize_samples(samples, limit=2, try_hard=False) is samples


def test_completion_uses_distinct_reproducible_seed_per_trial():
    designer = SAFEDesign.__new__(SAFEDesign)
    designer.verbose = False
    designer.safe_encoder = MagicMock()
    designer.safe_encoder.slicer = None
    designer.safe_encoder.encoder.return_value = "CC"
    designer._generate = MagicMock(return_value=["CC"])
    designer._decode_safe = MagicMock(side_effect=lambda values, **_: values)

    designer._completion("CC", n_samples_per_trial=1, n_trials=3, random_seed=11)

    seeds = [call.kwargs["seed"] for call in designer.safe_encoder.encoder.call_args_list]
    expected_rng = random.Random(11)
    expected = [expected_rng.randint(1, 2**32 - 1) for _ in range(3)]
    assert seeds == expected
    assert len(set(seeds)) == 3


def test_default_model_load_is_revision_pinned(monkeypatch):
    model = MagicMock(spec=SAFEDoubleHeadsModel)
    model.config = SimpleNamespace()
    tokenizer = MagicMock(spec=SAFETokenizer)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    generation_config = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0)

    model_loader = MagicMock(return_value=model)
    tokenizer_loader = MagicMock(return_value=tokenizer)
    config_loader = MagicMock(return_value=generation_config)
    monkeypatch.setattr(SAFEDoubleHeadsModel, "from_pretrained", model_loader)
    monkeypatch.setattr(SAFETokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(GenerationConfig, "from_pretrained", config_loader)

    SAFEDesign.load_default()

    expected = {"revision": SAFEDesign._DEFAULT_MODEL_REVISION}
    model_loader.assert_called_once_with(SAFEDesign._DEFAULT_MODEL_PATH, **expected)
    tokenizer_loader.assert_called_once_with(SAFEDesign._DEFAULT_MODEL_PATH, **expected)
    config_loader.assert_called_once_with(SAFEDesign._DEFAULT_MODEL_PATH, **expected)
