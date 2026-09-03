from collections import Counter
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import datamol as dm
import numpy as np
import pytest
from rdkit import Chem

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


def test_canonical_encoder_is_invariant_to_equivalent_smiles_and_randomize_flag():
    smiles = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    mol = dm.to_mol(smiles)
    equivalent_smiles = [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(10)]

    encodings = {safe.encode(item, canonical=True) for item in equivalent_smiles}

    assert len(encodings) == 1
    assert safe.encode(smiles, canonical=True, randomize=True, seed=42) in encodings


def test_noncanonical_encoding_does_not_depend_on_python_hash_seed():
    code = (
        "import safe; "
        "print(safe.SAFEConverter(slicer=None).encoder("
        "'[*]CC.[*]N', canonical=False, allow_empty=True))"
    )
    outputs = []
    for hash_seed in ("0", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1]


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


def test_safe_decoder_has_explicit_strict_and_permissive_paths():
    invalid_safe = "C(=2)c1ccccc1.c13ccccc1.C(=2)CC3"

    assert safe.decode(invalid_safe, ignore_errors=True) is None
    with pytest.raises(safe.SAFEDecodeError):
        safe.decode(invalid_safe, ignore_errors=False)


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
    # Stereo-sensitive double bonds are no longer cut, so both serialization
    # modes remain parseable and preserve the complete isomer.
    assert dm.same_mol(input_sm, safe.decode(failing_encoded))
    assert dm.same_mol(input_sm, safe.decode(working_encoded))


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
        converter = safe.SAFEConverter(slicer="rotatable", ignore_stereo=False)
        output_string = converter.encoder(mol, canonical=True, allow_empty=True)
        assert dm.same_mol(mol, output_string)

    # Stereogenic bonds are not cut: SAFE must preserve E/Z.
    converter = safe.SAFEConverter(slicer="brics", ignore_stereo=False)
    output = converter.encoder(STEREO_MOL_LIST[0], canonical=True, allow_empty=True)
    assert dm.same_mol(STEREO_MOL_LIST[0], safe.decode(output))

    # check if we ignore the stereo
    output = safe.encode(STEREO_MOL_LIST[0], ignore_stereo=True, slicer="brics")
    assert dm.same_mol(dm.remove_stereochemistry(dm.to_mol(STEREO_MOL_LIST[0])), output)


@pytest.mark.parametrize(
    "smiles",
    [
        "F/C=C/F",
        "F/C=C\\F",
        "F/C=C/C=C/F",
        r"C=C1/C(=C\C=C2/CCC[C@]3(C)[C@@H]([C@H](C)/C=C/[C@@H](O)C4CC4)CC[C@@H]23)C[C@@H](O)C[C@@H]1O",
        r"CC1=C(/C=C/C(C)=C\C=C\C(C)=C\C(=O)O)C(C)(C)CCC1",
        r"COc1cc(C)c(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)c(C)c1C",
        "N[C@@H](C)C(=O)O",
        "C[C@H](O)[C@@H](N)C(=O)O",
        r"C(=C/c1ccccc1)\CCc1ccccc1",
    ],
)
@pytest.mark.parametrize("slicer", ["brics", "hr", "recap", "mmpa", "rotatable"])
def test_stereochemistry_round_trip_across_slicers(smiles, slicer):
    converter = safe.SAFEConverter(slicer=slicer, ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert dm.same_mol(smiles, converter.decoder(encoded))


def test_directional_single_bonds_can_be_cut_around_one_stereogenic_double_bond():
    smiles = "F/C=C/F"

    def directional_single_bonds(mol):
        return [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.SINGLE and bond.GetBondDir() != Chem.BondDir.NONE
        ]

    converter = safe.SAFEConverter(slicer=directional_single_bonds, ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True)

    assert encoded.count(".") == 2
    assert dm.same_mol(smiles, converter.decoder(encoded))


def test_single_bond_shared_by_two_stereogenic_double_bonds_is_not_cut():
    smiles = "F/C=C/C=C/F"

    def shared_single_bond(mol):
        return [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.SINGLE
            and all(
                any(
                    adjacent.GetBondType() == Chem.BondType.DOUBLE
                    and adjacent.GetStereo()
                    not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
                    for adjacent in atom.GetBonds()
                )
                for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
            )
        ]

    converter = safe.SAFEConverter(slicer=shared_single_bond, ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert "." not in encoded
    assert dm.same_mol(smiles, converter.decoder(encoded))


def test_attach_cuts_heavy_references_but_keeps_explicit_stereo_hydrogens():
    smiles = "F/C=C/F"
    converter = safe.SAFEConverter(slicer="attach", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert encoded.count(".") == 2
    assert dm.same_mol(smiles, converter.decoder(encoded))


@pytest.mark.parametrize(
    "smiles",
    [
        r"C/C=C(/C)C(=O)O[C@H]1C(C)=C[C@]23C(=O)[C@@H](C=C(CO)[C@@H](O)[C@]12O)[C@H]1[C@@H](C[C@H]3C)C1(C)C",
        r"C1=C/COCc2cc(ccc2OCCN2CCCC2)Nc2nccc(n2)-c2cccc(c2)COC/1.O=C(O)CC(O)(CC(=O)O)C(=O)O",
    ],
)
def test_attach_preserves_ez_with_explicit_hydrogen_fragmentation(smiles):
    converter = safe.SAFEConverter(slicer="attach", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert dm.same_mol(smiles, converter.decoder(encoded))


def test_noncarbon_stereocenter_bonds_are_not_cut():
    smiles = "C[S@+](CC)[O-]"
    converter = safe.SAFEConverter(slicer="rotatable", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert "." not in encoded
    assert dm.same_mol(smiles, converter.decoder(encoded))


@pytest.mark.parametrize(
    "smiles",
    [
        "C[S@+](CC)[O-]",
        "CO[P@@]1OCC2CCC[C@@H]2O1",
    ],
)
def test_public_encode_falls_back_when_every_cut_is_stereo_unsafe(smiles):
    encoded = safe.encode(smiles, slicer="rotatable")

    assert "." not in encoded
    assert dm.same_mol(smiles, safe.decode(encoded))


@pytest.mark.parametrize(
    "smiles",
    [
        "S[As@TB1](F)(Cl)(Br)N",
        "O=[Co@OH1](Cl)(F)(I)(Br)S",
        "Cl[Pt@SP1](Cl)(N)N",
    ],
)
def test_non_tetrahedral_stereo_is_not_fragmented_or_changed(smiles):
    converter = safe.SAFEConverter("hr", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True)
    decoded = converter.decoder(encoded, canonical=True)
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)

    assert "." not in encoded
    assert decoded == expected


@pytest.mark.parametrize("group", ["&1", "o1", "a"])
def test_enhanced_stereo_groups_are_rejected_instead_of_silently_lost(group):
    smiles = f"C[C@H](O)[C@H](F)Cl |{group}:1,3|"

    with pytest.raises(safe.SAFEEncodeError, match="Enhanced CXSMILES stereo groups"):
        safe.encode(smiles)

    converter = safe.SAFEConverter(slicer=None, ignore_stereo=True)
    assert converter.encoder(smiles, allow_empty=True)


def test_attach_keeps_hydrogen_cuts_away_from_stereocenters():
    smiles = (
        "CC(=O)O[C@@H]1C[C@@H]2C[C@]3(CC[C@]2(C)[C@H]2CC[C@]4(C)"
        "[C@@H]([C@H](C)CCC(=O)O)CC[C@H]4[C@H]12)OOC1(CC[C@@H](C)CC1)OO3"
    )
    converter = safe.SAFEConverter(slicer="attach", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert "." in encoded
    assert dm.same_mol(smiles, converter.decoder(encoded))


@pytest.mark.parametrize("canonical", [True, False])
def test_attach_falls_back_before_changing_constrained_peroxide_stereo(canonical):
    smiles = (
        "CC[C@H]1CCC2(CC1)OO[C@]1(CC[C@@]3(C)[C@H](C[C@@H](OC(C)=O)"
        "[C@@H]4[C@@H]3CC[C@]3(C)[C@@H]([C@H](C)CCC(=O)OC)CC[C@@H]43)C1)OO2"
    )
    converter = safe.SAFEConverter(slicer="attach", ignore_stereo=False)
    encoded = converter.encoder(
        smiles,
        canonical=canonical,
        randomize=not canonical,
        seed=0,
        allow_empty=True,
    )

    assert converter._canonical_isomeric_graph(
        converter.decoder(encoded)
    ) == converter._canonical_isomeric_graph(smiles)


def test_attach_preserves_ez_while_cutting_remote_bonds():
    smiles = r"Cn1c2c(c(=O)n(C)c1=O)/N=N\c1c(c(=O)n(C)c(=O)n1C)/N=N\2"
    converter = safe.SAFEConverter(slicer="attach", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert "." in encoded
    assert dm.same_mol(smiles, converter.decoder(encoded))


def test_canonical_decode_does_not_standardize_charge_or_tautomer():
    smiles = "CCOC(=O)[N-]c1c[n+](N2CCOCC2)no1"
    encoded = safe.SAFEConverter("brics").encoder(smiles, canonical=True, allow_empty=True)

    assert dm.same_mol(smiles, safe.decode(encoded, canonical=True))


def test_attr_as_restores_value_after_exception():
    obj = SimpleNamespace(value="before")

    with pytest.raises(RuntimeError), safe.utils.attr_as(obj, "value", "during"):
        assert obj.value == "during"
        raise RuntimeError("test")

    assert obj.value == "before"


def test_mol_slicer_returns_no_linker_when_minimum_size_cannot_be_met():
    mol = dm.to_mol("c1ccccc1CCc1ccccc1")
    head, linker, tail = safe.utils.MolSlicer(
        min_linker_size=100,
        require_ring_system=False,
    )(mol)

    assert dm.same_mol(head, mol)
    assert linker is None
    assert tail is None


def test_convert_to_safe_accepts_molecule_without_string_membership_check():
    mol = dm.to_mol("CC")

    result = safe.utils.convert_to_safe(mol, split_fragment=True)

    assert result is None or isinstance(result, str)


def test_substructure_filter_skips_invalid_molecules_and_rejects_invalid_query():
    assert safe.utils.filter_by_substructure_constraints(["CC", "not-smiles"], "C") == ["CC"]

    with pytest.raises(ValueError, match="Substructure constraint"):
        safe.utils.filter_by_substructure_constraints(["CC"], "[invalid")


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


def test_large_polymer_round_trip_uses_extended_ring_closures():
    # A synthetic polymer (nylon-6 oligomer) is fragmented at every amide bond,
    # producing far more than 99 SAFE fragments. Their attachment bonds must use
    # the %(nnn) extended ring-closure form and still round-trip exactly.
    n_units = 130
    polymer = "NCCCCC" + "C(=O)NCCCCC" * (n_units - 1) + "C(=O)O"

    encoded = safe.encode(polymer, canonical=True)

    assert encoded.count(".") + 1 > 99  # many fragments
    assert "%(" in encoded  # uses extended ring closures
    assert dm.same_mol(polymer, safe.decode(encoded, canonical=True))


def test_directional_extended_ring_closures_are_standardized():
    component = r"CC/C=C\C/C=C\C/C=C\C/C=C\C/C=C\C/C=C\CCC(=O)O"
    smiles = ".".join([component] * 7)

    converter = safe.SAFEConverter("hr", ignore_stereo=False)
    encoded = converter.encoder(smiles, canonical=True, allow_empty=True)

    assert re.search(r"[/\\]%\(\d+\)", encoded)
    assert not re.search(r"\([/\\]%\(\d+\)\)", encoded)
    assert dm.same_mol(smiles, converter.decoder(encoded))


def test_extended_ring_closure_tokenization():
    assert safe.split("C%(1).N%(12).O%(100).S%99") == [
        "C",
        "%(1)",
        ".",
        "N",
        "%(12)",
        ".",
        "O",
        "%(100)",
        ".",
        "S",
        "%99",
    ]


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
        decoded = converter.decoder(encoded, remove_dummies=False, canonical=True)
        assert decoded.count("*") == 2


@pytest.mark.parametrize("wildcard", ["C[*:1]C", "C[1*]C"])
def test_internal_wildcards_keep_their_topology(wildcard):
    converter = safe.SAFEConverter(slicer=None)
    encoded = converter.encoder(wildcard, canonical=True, allow_empty=True)
    decoded = converter.decoder(encoded, remove_dummies=False, canonical=True)

    assert "*" in encoded
    assert decoded == "C*C"


@pytest.mark.parametrize("wildcard", ["[*:1]", "[1*]"])
def test_lone_wildcards_do_not_crash_encoding(wildcard):
    converter = safe.SAFEConverter(slicer=None)
    encoded = converter.encoder(wildcard, canonical=True, allow_empty=True)

    assert encoded == "*"
    assert converter.decoder(encoded, remove_dummies=False, canonical=True) == "*"


def test_legacy_fragment_override_keeps_working():
    class LegacyConverter(safe.SAFEConverter):
        def _fragment(self, mol, allow_empty=False):
            del mol, allow_empty
            return []

    assert LegacyConverter().encoder("CC", allow_empty=True) == "CC"


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # aromatic ring, no BRICS bond
        "C1CCCCC1",  # saturated ring, no BRICS bond
        "C",  # single atom
        "[Na+].[Cl-]",  # salt whose components cannot be cut
        "CC(=O)[O-].[Na+]",  # carboxylate salt
        "CCO.O",  # multi-component (solvate)
    ],
)
def test_encode_unbreakable_molecule_raises_by_default(smiles):
    with pytest.raises(safe.SAFEFragmentationError):
        safe.encode(smiles)


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",
        "C1CCCCC1",
        "C",
        "[Na+].[Cl-]",
        "CC(=O)[O-].[Na+]",
        "CCO.O",
    ],
)
def test_encode_allow_empty_returns_unfragmented_and_round_trips(smiles):
    encoded = safe.encode(smiles, allow_empty=True)
    # An uncuttable input becomes a single SAFE block: no fragment separator
    # is introduced beyond the disconnections already present in the input.
    assert encoded.count(".") == smiles.count(".")
    assert dm.same_mol(smiles, encoded)
    assert dm.same_mol(smiles, safe.decode(encoded, canonical=True))


@pytest.mark.parametrize(
    "scaffold,n_attachments",
    [
        ("O=c1[nH]cnc2nc([*])ccc12", 1),
        ("c1ccc([*])cc1", 1),
        ("c1cc([*])ccc1[*]", 2),
    ],
)
def test_scaffold_attachment_points_survive_round_trip(scaffold, n_attachments):
    """Regression test for https://github.com/datamol-io/safe/issues/67."""
    converter = safe.SAFEConverter(slicer=None)
    encoded = converter.encoder(scaffold, canonical=True, allow_empty=True)
    decoded = converter.decoder(encoded, remove_dummies=False, canonical=True)

    assert encoded.count("*") == 0
    assert decoded.count("*") == n_attachments
