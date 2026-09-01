from collections import Counter

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
        "C1*CCC1COO",
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


def test_fused_ring_issue():
    FUSED_RING_LIST = [
        "[H][C@@]12CC[C@@]3(CCC(=O)O3)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])[C@@]([H])(CC2=CC(=O)CC[C@]12C)SC(C)=O",
        "[H][C@@]12C[C@H](C)[C@](OC(=O)CC)(C(=O)COC(=O)CC)[C@@]1(C)C[C@H](O)[C@@]1(Cl)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C",
        "[H][C@@]12CC[C@@](O)(C#C)[C@@]1(CC)CC[C@]1([H])[C@@]3([H])CCC(=O)C=C3CC[C@@]21[H]",
    ]
    for fused_ring in FUSED_RING_LIST:
        output_string = safe.decode(safe.encode(fused_ring))
        assert dm.same_mol(fused_ring, output_string)


def test_stereochemistry_issue():
    STEREO_MOL_LIST = [
        "CC(=C\\c1ccccc1)/N=C/C(=O)O",
        "CC(=C/c1ccccc1)/N=C/C(=O)O",
        "CC(=C\\c1ccccc1)/N=C\\C(=O)O",
        "CC(=C/c1ccccc1)/N=C\\C(=O)O",
        "CC(=Cc1ccccc1)N=CC(=O)O",
        "Cc1ccc(-n2c(C)cc(/C=N/Nc3ccc([N+](=O)[O-])cn3)c2C)c(C)c1",
        "Cc1ccc(-n2c(C)cc(/C=N\\Nc3ccc([N+](=O)[O-])cn3)c2C)c(C)c1",
    ]
    for mol in STEREO_MOL_LIST:
        output_string = safe.encode(mol, ignore_stereo=False, slicer="rotatable")
        assert dm.same_mol(mol, output_string)

    # now let's test failure case where we fail because we split on a double bond
    output = safe.encode(STEREO_MOL_LIST[0], ignore_stereo=False, slicer="brics")
    assert dm.same_mol(STEREO_MOL_LIST[0], output) is False
    same_stereo = [dm.remove_stereochemistry(dm.to_mol(x)) for x in [output, STEREO_MOL_LIST[0]]]
    assert dm.same_mol(same_stereo[0], same_stereo[1])

    # check if we ignore the stereo
    output = safe.encode(STEREO_MOL_LIST[0], ignore_stereo=True, slicer="brics")
    assert dm.same_mol(dm.remove_stereochemistry(dm.to_mol(STEREO_MOL_LIST[0])), output)


def test_large_molecule_ring_closures():
    # A long peptide produces > 99 SAFE fragments, whose attachment bonds need
    # the %(nnn) extended ring-closure form to be valid SMILES.
    from rdkit import Chem

    seq = "LVYTDCTESGQNLCLCEGSNVCGQGNKCILGSDGEKNQCVTGEGTPKPQSHNDGDFEEIPEEYLQ"
    smiles = Chem.MolToSmiles(Chem.MolFromSequence(seq))
    encoded = safe.encode(smiles, canonical=True)
    assert "%(" in encoded  # uses extended ring closures
    assert dm.same_mol(smiles, safe.decode(encoded))


def test_extended_ring_closure_decoding():
    # The decoder must understand RDKit's extended '%(nnn)' ring-closure form,
    # both when reading branch numbers and when completing unpaired attachment
    # points.
    from rdkit import Chem

    conv = safe.SAFEConverter()

    # '%(nnn)' is a single ring-closure label, not three separate digits
    assert conv._find_branch_number("C%(100)") == [100]
    assert conv._find_branch_number("c1ccccc1%(123)") == [1, 1, 123]
    # plain single-digit and two-digit forms keep their existing behaviour
    assert conv._find_branch_number("C1CC%23C") == [1, 23]
    # isotope and atom-map digits inside brackets are not ring closures
    assert conv._find_branch_number("[13C:2]C12CC%23C%(100)") == [1, 2, 23, 100]

    # an unpaired extended label must be completed into a valid molecule
    assert dm.to_mol(conv._ensure_valid("C%(100)")) is not None

    # fragment-level decoding of a >99 ring-closure molecule must not silently fail
    seq = "LVYTDCTESGQNLCLCEGSNVCGQGNKCILGSDGEKNQCVTGEGTPKPQSHNDGDFEEIPEEYLQ"
    encoded = safe.encode(Chem.MolToSmiles(Chem.MolFromSequence(seq)), canonical=True)
    decoded_fragments = [safe.decode(fragment, fix=True) for fragment in encoded.split(".")]
    assert all(x is not None for x in decoded_fragments)
    assert safe.decode(encoded, as_mol=True) is not None


def test_extended_ring_closure_tokenization():
    assert safe.split("C%(100).N%99") == ["C", "%(100)", ".", "N", "%99"]


def test_explicit_attachment_points_remain_open():
    converter = safe.SAFEConverter(slicer=None)
    for side_chains in ("[1*]CC.[2*]N", "[*]CC.[*]N"):
        encoded = converter.encoder(
            side_chains,
            canonical=False,
            allow_empty=True,
        )

        branch_counts = Counter(converter._find_branch_number(encoded))
        assert len([label for label, count in branch_counts.items() if count % 2]) == 2
        assert "*" not in encoded
