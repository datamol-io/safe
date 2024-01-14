from safe.sample import SAFEDesign
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel


def test_load_default_safe_model():
    model = SAFEDoubleHeadsModel.from_pretrained("datamol-io/safe-gpt")
    assert model is not None
    assert isinstance(model, SAFEDoubleHeadsModel)


def test_load_default_safe_tokenizer():
    tokenizer = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")
    assert isinstance(tokenizer, SAFETokenizer)


def test_check_molecule_sampling():
    designer = SAFEDesign.load_default(verbose=True)
    generated = designer.de_novo_generation(sanitize=True, n_samples_per_trial=10)
    assert len(generated) > 0
