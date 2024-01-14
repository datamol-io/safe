import datamol as dm
import numpy as np
import pytest

import safe


def test_safe_encoding():
    celecoxib = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    expected_encodings = "c13ccc(S(N)(=O)=O)cc1.Cc1ccc4cc1.c14cc5nn13.C5(F)(F)F"
    safe_celecoxib = safe.encode(celecoxib, canonical=True)
    dec_celecoxib = safe.decode(safe_celecoxib)
    assert safe_celecoxib.count(".") == 3  # 3 fragments
    # we compare length since digits can be random
    assert len(safe_celecoxib) == len(expected_encodings)
    assert dm.same_mol(celecoxib, safe_celecoxib)
    assert dm.same_mol(celecoxib, dec_celecoxib)


def test_safe_fragment_randomization():
    celecoxib = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    safe_celecoxib = safe.encode(celecoxib)
    fragments = safe_celecoxib.split(".")
    randomized_fragment_safe_str = np.random.permutation(fragments).tolist()
    randomized_fragment_safe_str = ".".join(randomized_fragment_safe_str)
    assert dm.same_mol(celecoxib, randomized_fragment_safe_str)


def test_randomized_encoder():
    celecoxib = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    output = set()
    for i in range(5):
        out = safe.encode(celecoxib, canonical=False, randomize=True, seed=i)
        output.add(out)
    assert len(output) > 1


def test_custom_encoder():
    smart_slicer = ["[r]-;!@[r]"]
    celecoxib = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    safe_str = safe.encode(celecoxib, canonical=True, slicer=smart_slicer)
    assert dm.same_mol(celecoxib, safe_str)


def test_safe_decoder():
    celecoxib = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    safe_str = safe.encode(celecoxib)
    fragments = safe_str.split(".")
    decoded_fragments = [safe.decode(x, fix=True) for x in fragments]
    assert [dm.to_mol(x) for x in fragments] == [None] * len(fragments)
    assert all(x is not None for x in decoded_fragments)


def test_rdkit_smiles_parser_issues():
    # see https://github.com/datamol-io/safe/issues/22
    input_sm = r"C(=C/c1ccccc1)\CCc1ccccc1"
    slicer = "brics"
    safe_obj = safe.SAFEConverter(slicer=slicer, require_hs=False)
    with dm.without_rdkit_log():
        failing_encoded = safe_obj.encoder(
            input_sm,
            canonical=True,
            randomize=False,
            rdkit_safe=False,
        )
        working_encoded = safe_obj.encoder(
            input_sm,
            canonical=True,
            randomize=False,
            rdkit_safe=True,
        )
    working_decoded = safe.decode(working_encoded)
    working_no_stero = dm.remove_stereochemistry(dm.to_mol(input_sm))
    input_mol = dm.remove_stereochemistry(dm.to_mol(working_decoded))
    assert safe.decode(failing_encoded) is None
    assert working_decoded is not None
    assert dm.same_mol(working_no_stero, input_mol)


@pytest.mark.parametrize(
    "input_sm",
    [
        "O=C(CN1CC[NH2+]CC1)N1CCCCC1",
        "[NH3+]Cc1ccccc1",
        "c1cc2c(cc1[C@@H]1CCC[NH2+]1)OCCO2",
        "[13C]1CCCCC1C[238U]C[NH3+]",
        "COC[CH2:1][CH2:2]O[CH:2]C[OH:3]",
    ],
)
def test_bracket_smiles_issues(input_sm):
    slicer = "brics"
    safe_obj = safe.SAFEConverter(slicer=slicer, require_hs=False)
    fragments = []
    with dm.without_rdkit_log():
        safe_str = safe_obj.encoder(
            input_sm,
            canonical=True,
        )
        for fragment in safe_str.split("."):
            f = safe_obj.decoder(
                fragment,
                as_mol=False,
                canonical=True,
                fix=True,
                remove_dummies=True,
                remove_added_hs=True,
            )
            fragments.append(f)
    input_mol = dm.to_mol(input_sm)
    assert safe.decode(safe_str) is not None
    assert dm.same_mol(dm.to_mol(safe_str), input_mol)
    assert None not in fragments
