import pytest
import torch
import transformers

from safe.sample import SAFEDesign
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def designer():
    return SAFEDesign.load_default(verbose=False, device="cpu")


def test_load_default_safe_model(designer):
    assert isinstance(designer.model, SAFEDoubleHeadsModel)


def test_load_default_safe_tokenizer(designer):
    assert isinstance(designer.tokenizer, SAFETokenizer)


def test_check_molecule_sampling(designer):
    generated = designer.de_novo_generation(sanitize=True, n_samples_per_trial=10)
    assert len(generated) > 0


def test_safe_gpt_logits_are_unchanged(designer):
    tokenizer = designer.tokenizer.get_pretrained()
    inputs = tokenizer("C.C", return_tensors="pt")
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = designer.model(**inputs, use_cache=False).logits

    expected = torch.tensor(
        [
            -2.0939908,
            -6.11428738,
            -1.89603281,
            -10.79415512,
            -10.77486801,
            -2.61483884,
            -10.87054348,
            -4.48864126,
            -1.88843548,
            -10.80654049,
        ]
    )
    torch.testing.assert_close(logits[0, -1, :10].cpu(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "name,kwargs,expected",
    [
        (
            "greedy",
            {"how": "greedy"},
            ["CCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCC"],
        ),
        (
            "multinomial",
            {"how": "random"},
            ["CC4.n14cc7c8n1.", "C[C@H]2CC(=O)[O-].N32."],
        ),
        (
            "beam",
            {"how": "beam", "num_beams": 4},
            ["Cc1ccc(C)c5c1.C", "Cc1ccc(C)c5c1.N"],
        ),
        (
            "beam sampling",
            {"how": "beam", "num_beams": 4, "do_sample": True},
            ["CCCCCCCCCCCCCCC", "Cc1ccc5cc1.C5CC"],
        ),
        (
            "diverse beam",
            {
                "how": "beam",
                "num_beams": 4,
                "num_beam_groups": 2,
                "diversity_penalty": 1.0,
            },
            ["Cc1ccc(C)c5c1.C", "Cc1ccc(C)c5c1.N"],
        ),
        (
            "constrained beam",
            {
                "how": "beam",
                "num_beams": 4,
                "force_words_ids": [[[11, 27], [11, 28]]],
            },
            ["Cc1ccccc15.C65.", "Cc1ccccc15.C5CC"],
        ),
    ],
)
def test_sampling_matches_transformers_4_baseline_with_and_without_cache(
    designer, name, kwargs, expected
):
    del name
    outputs = []
    for use_cache in (False, True):
        transformers.set_seed(123)
        outputs.append(
            designer._generate(
                n_samples=2,
                safe_prefix="C",
                max_length=16,
                use_cache=use_cache,
                **kwargs,
            )
        )

    assert outputs == [expected, expected]


def test_contrastive_search_is_reproducible(designer):
    """Contrastive search requires its cache and a pinned custom backend."""
    outputs = []
    for _ in range(2):
        transformers.set_seed(123)
        outputs.append(
            designer._generate(
                n_samples=2,
                safe_prefix="C",
                max_length=16,
                how=None,
                top_k=4,
                penalty_alpha=0.6,
                use_cache=True,
            )
        )

    assert outputs[0] == outputs[1]
    assert outputs[0] == ["C[C@@H]46.c16ccc(F)c", "C[C@@H]46.c16ccc(F)c"]


def test_prompt_lookup_assisted_sampling_requires_cache(designer):
    transformers.set_seed(123)
    generated = designer._generate(
        n_samples=2,
        safe_prefix="C",
        max_length=16,
        how="greedy",
        prompt_lookup_num_tokens=3,
        use_cache=True,
    )
    assert generated == ["CCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCC"]

    with pytest.raises(ValueError, match="requires `use_cache=True`"):
        designer._generate(
            n_samples=2,
            safe_prefix="C",
            max_length=16,
            how="greedy",
            prompt_lookup_num_tokens=3,
            use_cache=False,
        )


def test_model_only_linker_uses_complete_constraints_and_returns_only_smiles(designer):
    transformers.set_seed(123)
    generated = designer.linker_generation(
        "[*]CC",
        "[*]N",
        n_samples_per_trial=2,
        n_trials=1,
        random_seed=123,
        max_length=48,
        model_only=True,
    )

    # Regression oracle produced by both Transformers 4.57.6 and 5.16.1.
    # Intermediate SAFE strings must not leak into the public result list.
    assert generated == ["CCC1CN1.N", "CCCCCN"]
