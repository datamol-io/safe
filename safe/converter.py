import itertools
import re
from collections import Counter
from typing import Callable, List, Optional, Union

import datamol as dm
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

from ._exception import SAFEDecodeError, SAFEEncodeError, SAFEFragmentationError


class SAFEConverter:
    """Molecule line notation conversion from SMILES to SAFE

    A SAFE representation is a string based representation of a molecule decomposition into fragment components,
    separated by a dot ('.'). Note that each component (fragment) might not be a valid molecule by themselves,
    unless explicitely correct to add missing hydrogens.

    !!! note "Slicing algorithms"

        By default SAFE strings are generated using `BRICS`, however, the following alternative are supported:

        * [Hussain-Rea (`hr`)](https://pubs.acs.org/doi/10.1021/ci900450m)
        * [RECAP (`recap`)](https://pubmed.ncbi.nlm.nih.gov/9611787/)
        * [RDKit's MMPA (`mmpa`)](https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html)
        * Any possible attachment points (`attach`)

        Furthermore, you can also provide your own slicing algorithm, which should return a pair of atoms
        corresponding to the bonds to break.

    """

    SUPPORTED_SLICERS = ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]
    __SLICE_SMARTS = {
        "hr": ["[*]!@-[*]"],  # any non ring single bond
        "recap": [
            "[$([C;!$(C([#7])[#7])](=!@[O]))]!@[$([#7;+0;!D1])]",
            "[$(C=!@O)]!@[$([O;+0])]",
            "[$([N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*]))]-!@[$([*])]",
            "[$(C(=!@O)([#7;+0;D2,D3])!@[#7;+0;D2,D3])]!@[$([#7;+0;D2,D3])]",
            "[$([O;+0](-!@[#6!$(C=O)])-!@[#6!$(C=O)])]-!@[$([#6!$(C=O)])]",
            "C=!@C",
            "[N;+1;D4]!@[#6]",
            "[$([n;+0])]-!@C",
            "[$([O]=[C]-@[N;+0])]-!@[$([C])]",
            "c-!@c",
            "[$([#7;+0;D2,D3])]-!@[$([S](=[O])=[O])]",
        ],
        "mmpa": ["[#6+0;!$(*=,#[!#6])]!@!=!#[*]"],  # classical mmpa slicing smarts
        "attach": ["[*]!@[*]"],  # any potential attachment point, including hydrogens when explicit
        "rotatable": ["[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"],
    }

    def __init__(
        self,
        slicer: Optional[Union[str, List[str], Callable]] = "brics",
        require_hs: Optional[bool] = None,
        use_original_opener_for_attach: bool = True,
        ignore_stereo: bool = False,
    ):
        """Constructor for the SAFE converter

        Args:
            slicer: slicer algorithm to use for encoding.
                Can either be one of the supported slicing algorithm (SUPPORTED_SLICERS)
                or a custom callable that returns the bond ids that can be sliced.
            require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added.
                `attach` slicer requires adding hydrogens.
            use_original_opener_for_attach: whether to use the original branch opener digit when adding back
                mapping number to attachment points, or use simple enumeration.
            ignore_stereo: whether to discard input stereochemistry explicitly. When false,
                stereochemistry-changing cuts are skipped and the encoded graph is verified.

        """
        self.slicer = slicer
        if isinstance(slicer, str) and slicer.lower() in self.SUPPORTED_SLICERS:
            self.slicer = self.__SLICE_SMARTS.get(slicer.lower(), slicer)
        if self.slicer != "brics" and isinstance(self.slicer, str):
            self.slicer = [self.slicer]
        if isinstance(self.slicer, (list, tuple)):
            self.slicer = [dm.from_smarts(x) for x in self.slicer]
            if any(x is None for x in self.slicer):
                raise ValueError(f"Slicer: {slicer} cannot be valid")
        self.require_hs = require_hs or (slicer == "attach")
        self.use_original_opener_for_attach = use_original_opener_for_attach
        self.ignore_stereo = ignore_stereo

    @staticmethod
    def randomize(mol: dm.Mol, rng: Optional[int] = None):
        """Randomize the position of the atoms in a mol.

        Args:
            mol: molecules to randomize
            rng: optional seed to use
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()
        if mol.GetNumAtoms() == 0:
            return mol
        atom_indices = list(range(mol.GetNumAtoms()))
        atom_indices = rng.permutation(atom_indices).tolist()
        return Chem.RenumberAtoms(mol, atom_indices)

    @staticmethod
    def _format_ring_closure(num: int) -> str:
        """Format a ring-closure (branch) number into its SMILES token.

        Single digits stay bare (e.g. ``5``), two-digit numbers use the ``%NN``
        form, and numbers >= 100 use RDKit's extended ``%(nnn)`` ring-closure
        notation (see https://www.rdkit.org/docs/RDKit_Book.html#ring-closures).

        Args:
            num: ring-closure number to format
        """
        if num < 10:
            return str(num)
        if num < 100:
            return f"%{num}"
        return f"%({num})"

    @classmethod
    def _find_branch_number_positions(cls, inp: str):
        """Find ring-closure labels and their positions in a SMILES string."""
        matches = re.finditer(r"\[[^\]]+\]|%\((\d+)\)|%(\d{2})|(\d+)", inp)
        branch_numbers = []
        for match in matches:
            extended, double, singles = match.groups()
            if extended:
                branch_numbers.append((int(extended), match.start()))
            elif double:
                branch_numbers.append((int(double), match.start()))
            elif singles:
                branch_numbers.extend(
                    (int(digit), match.start(3) + offset) for offset, digit in enumerate(singles)
                )
        return branch_numbers

    @classmethod
    def _find_branch_number(cls, inp: str):
        """Find the branch numbers and ring closures in a SMILES representation.

        Args:
            inp: input smiles
        """
        return [label for label, _ in cls._find_branch_number_positions(inp)]

    def _ensure_valid(self, inp: str):
        """Ensure that the input SAFE string is valid by fixing the missing attachment points

        Args:
            inp: input SAFE string

        """
        missing_tokens = [inp]
        branch_numbers = self._find_branch_number(inp)
        # only use the set that have exactly 1 element
        # any branch number that is not pairwise should receive a dummy atom to complete the attachment point
        branch_numbers = Counter(branch_numbers)
        for i, (bnum, bcount) in enumerate(branch_numbers.items()):
            if bcount % 2 != 0:
                bnum_str = self._format_ring_closure(bnum)
                _tk = f"[*:{i+1}]{bnum_str}"
                if self.use_original_opener_for_attach:
                    _tk = f"[*:{bnum}]{bnum_str}"
                missing_tokens.append(_tk)
        return ".".join(missing_tokens)

    def decoder(
        self,
        inp: str,
        as_mol: bool = False,
        canonical: bool = False,
        fix: bool = True,
        remove_dummies: bool = True,
        remove_added_hs: bool = True,
    ):
        """Convert input SAFE representation to smiles

        Args:
            inp: input SAFE representation to decode as a valid molecule or smiles
            as_mol: whether to return a molecule object or a smiles string
            canonical: whether to return a canonical
            fix: whether to fix the SAFE representation to take into account non-connected attachment points
            remove_dummies: whether to remove dummy atoms from the SAFE representation. Set this to
                ``False`` when decoding an open SAFE fragment or scaffold if attachment points
                must be preserved.
            remove_added_hs: whether to remove all the added hydrogen atoms after applying dummy removal for recovery
        """

        if fix:
            inp = self._ensure_valid(inp)
        mol = dm.to_mol(inp)
        if mol is None:
            raise ValueError("SAFE string could not be parsed into a molecule")
        if remove_dummies:
            dummy_query = dm.from_smarts("[$([#0]!-!:*);$([#0;D1])]")
            if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                replacements = Chem.ReplaceSubstructs(
                    mol,
                    dummy_query,
                    dm.to_mol("C"),
                    True,
                )
                mol = dm.remove_dummies(replacements[0])
        if as_mol:
            if remove_added_hs:
                mol = dm.remove_hs(mol, update_explicit_count=True)
            return mol
        out = dm.to_smiles(mol, canonical=canonical, explicit_hs=(not remove_added_hs))
        has_stereo = any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
        ) or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())
        mol_graph = self._canonical_isomeric_graph(mol)
        if has_stereo and self._canonical_isomeric_graph(out) != mol_graph:
            # RDKit's non-canonical writer can choose an inconsistent parity
            # for rare symmetry-dependent stereocentres. Canonical writing is
            # deterministic and preserves the graph in those cases.
            canonical_out = dm.to_smiles(
                mol,
                canonical=True,
                explicit_hs=(not remove_added_hs),
            )
            out = (
                canonical_out if self._canonical_isomeric_graph(canonical_out) == mol_graph else inp
            )
        return out

    @staticmethod
    def _canonical_isomeric_graph(inp: Union[str, dm.Mol]):
        """Return a map-independent, dummy-aware isomeric graph identity."""
        mol = dm.to_mol(inp, remove_hs=False)
        if mol is None:
            return None
        mol = dm.remove_hs(mol, update_explicit_count=True)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def _fragment(self, mol: dm.Mol, allow_empty: bool = False):
        """
        Perform bond cutting in place for the input molecule, given the slicing algorithm

        Args:
            mol: input molecule to split
            allow_empty: whether to allow the slicing algorithm to return empty bonds
        Raises:
            SAFEFragmentationError: if the slicing algorithm return empty bonds
        """

        if self.slicer is None:
            matching_bonds = []

        elif callable(self.slicer):
            matching_bonds = self.slicer(mol)
            matching_bonds = list(matching_bonds)

        elif self.slicer == "brics":
            matching_bonds = BRICS.FindBRICSBonds(mol)
            matching_bonds = [brics_match[0] for brics_match in matching_bonds]

        else:
            matches = set()
            for smarts in self.slicer:
                matches |= {
                    tuple(sorted(match)) for match in mol.GetSubstructMatches(smarts, uniquify=True)
                }
            matching_bonds = list(matches)

        if matching_bonds is None or len(matching_bonds) == 0 and not allow_empty:
            raise SAFEFragmentationError(
                "Slicing algorithms did not return any bonds that can be cut !"
            )

        if not self.ignore_stereo:
            specified_stereos = [
                stereo
                for stereo in Chem.FindPotentialStereo(mol)
                if stereo.specified == Chem.StereoSpecified.Specified
            ]
            specified_double_bonds = {
                stereo.centeredOn
                for stereo in specified_stereos
                if stereo.type == Chem.StereoType.Bond_Double
            }
            specified_atom_centers = {
                stereo.centeredOn
                for stereo in specified_stereos
                if stereo.type
                in {
                    Chem.StereoType.Atom_Tetrahedral,
                    Chem.StereoType.Atom_SquarePlanar,
                    Chem.StereoType.Atom_TrigonalBipyramidal,
                    Chem.StereoType.Atom_Octahedral,
                }
            }

            # Explicit H cuts can alter rooted-fragment parity near an atom
            # stereocentre even when the cut atom itself is not stereogenic.
            # Build the protected two-hop neighbourhood once rather than doing
            # a shortest-path query for every candidate bond.
            atoms_near_stereocenters = set(specified_atom_centers)
            frontier = set(specified_atom_centers)
            if self.require_hs:
                for _ in range(2):
                    frontier = {
                        neighbor.GetIdx()
                        for atom_idx in frontier
                        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                        if neighbor.GetAtomicNum() != 1
                        and neighbor.GetIdx() not in atoms_near_stereocenters
                    }
                    atoms_near_stereocenters.update(frontier)

            # Cutting a specified double bond loses its E/Z metadata. A single
            # bond shared by two specified double bonds is unsafe for a subtler
            # reason: its SMILES direction participates in both local stereo
            # definitions, but the fragments are serialized independently
            # before their dummy atoms become one SAFE ring closure. Directional
            # single bonds belonging to only one double bond round-trip safely,
            # except for newly explicit hydrogen bonds: unlike the directional
            # heavy-atom reference, they carry no slash direction of their own.
            stereo_safe_bonds = []
            for atom_pair in matching_bonds:
                bond = mol.GetBondBetweenAtoms(*atom_pair)
                if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                    continue
                atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in atom_pair]
                adjacent_stereo_bonds = {
                    adjacent_bond.GetIdx()
                    for atom_idx in atom_pair
                    for adjacent_bond in mol.GetAtomWithIdx(atom_idx).GetBonds()
                    if adjacent_bond.GetIdx() in specified_double_bonds
                }
                cuts_explicit_stereo_hydrogen = (
                    self.require_hs
                    and bool(adjacent_stereo_bonds)
                    and any(atom.GetAtomicNum() == 1 for atom in atoms)
                )
                cuts_noncarbon_stereocenter = any(
                    atom_idx in specified_atom_centers and atom.GetAtomicNum() != 6
                    for atom_idx, atom in zip(atom_pair, atoms)
                )
                cuts_hydrogen_near_stereocenter = self.require_hs and any(
                    atom.GetAtomicNum() == 1
                    and any(
                        neighbor.GetIdx() in atoms_near_stereocenters
                        for neighbor in atom.GetNeighbors()
                    )
                    for atom in atoms
                )
                cuts_multiple_bond_with_ez = (
                    bool(specified_double_bonds) and bond.GetBondType() != Chem.BondType.SINGLE
                )
                if bond.GetBondType() == Chem.BondType.SINGLE and (
                    len(adjacent_stereo_bonds) > 1 or cuts_explicit_stereo_hydrogen
                ):
                    continue
                if (
                    cuts_noncarbon_stereocenter
                    or cuts_hydrogen_near_stereocenter
                    or cuts_multiple_bond_with_ez
                ):
                    continue
                stereo_safe_bonds.append(atom_pair)
            matching_bonds = stereo_safe_bonds

        # A slicer did find bonds, but every cut was unsafe for the specified
        # stereochemistry. Returning the molecule unfragmented is exact and is
        # preferable to rejecting an otherwise valid public ``safe.encode`` call.
        return matching_bonds or []

    def encoder(
        self,
        inp: Union[str, dm.Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[dm.Mol]] = None,
        allow_empty: bool = False,
        rdkit_safe: bool = True,
    ):
        """Convert input smiles to SAFE representation

        Args:
            inp: input smiles
            canonical: whether to return canonical smiles string. Defaults to True
            randomize: whether to randomize the safe string encoding. Will be ignored if canonical is provided
            seed: optional seed to use when allowing randomization of the SAFE encoding.
                Randomization happens at two steps:
                1. at the original smiles representation by randomization the atoms.
                2. at the SAFE conversion by randomizing fragment orders
            constraints: List of molecules or pattern to preserve during the SAFE construction. Any bond slicing would
                happen outside of a substructure matching one of the patterns.
            allow_empty: whether to allow the slicing algorithm to return empty bonds
            rdkit_safe: whether to apply rdkit-safe digit standardization to the output SAFE string.
        """
        source_text = inp if isinstance(inp, str) else None
        source_mol = dm.to_mol(inp, remove_hs=False)
        if source_mol is None:
            raise ValueError("Input could not be parsed into a molecule")
        if not self.ignore_stereo and source_mol.GetStereoGroups():
            raise SAFEEncodeError(
                "Enhanced CXSMILES stereo groups are not representable in SAFE 1.0; "
                "resolve them to a single stereoisomer or set ignore_stereo=True explicitly"
            )

        rng = None
        should_randomize = bool(randomize and not canonical)
        if should_randomize:
            rng = np.random.default_rng(seed)
            inp = self.randomize(source_mol, rng)

        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(inp, canonical=canonical, randomize=False, ordered=False)
        elif canonical:
            # Canonical SAFE must not depend on which equivalent SMILES spelling
            # the caller supplied. Molecule inputs already followed this path;
            # normalize string inputs before choosing rooted fragment atoms.
            inp = dm.to_smiles(source_mol, canonical=True, randomize=False, ordered=False)

        # EN: we first normalize the attachment if the molecule is a query:
        # inp = dm.reactions.convert_attach_to_isotope(inp, as_smiles=True)

        # RDKit's extended ring-closure form ('%(nnn)', up to 5 digits) is used for
        # labels >= 100; see `_format_ring_closure`.
        # https://www.rdkit.org/docs/RDKit_Book.html#ring-closures
        branch_numbers = self._find_branch_number(inp)

        mol = dm.to_mol(inp, remove_hs=False)
        if mol is None:
            raise ValueError("Input could not be parsed into a molecule")
        # Inspect explicit tags on the original graph. FindPotentialStereo can
        # omit symmetry-dependent tags in constrained peroxide systems, and
        # atom renumbering must never disable the final identity guard.
        has_specified_stereo = any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in source_mol.GetAtoms()
        ) or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in source_mol.GetBonds())
        if self.ignore_stereo:
            mol = dm.remove_stereochemistry(mol)

        bond_map_id = 1
        open_attachment_ids = set()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                # Preserve the distinction between an explicitly labelled
                # attachment point (for example ``[1*]`` or ``[*:1]``), or a
                # terminal ``[*]``, and a literal wildcard atom embedded in a
                # structure (for example ``C1*CCC1``). All are normalised below
                # so fragment labels remain unique, but only attachment points
                # must survive as unmatched SAFE ring closures for constrained
                # generation.
                if atom.GetDegree() == 1:
                    open_attachment_ids.add(bond_map_id)
                atom.SetAtomMapNum(0)
                atom.SetIsotope(bond_map_id)
                bond_map_id += 1

        if self.require_hs:
            mol = dm.add_hs(mol)
        matching_bonds = self._fragment(mol, allow_empty=allow_empty)
        substructed_ignored = []
        if constraints is not None:
            substructed_ignored = list(
                itertools.chain(
                    *[
                        mol.GetSubstructMatches(constraint, uniquify=True)
                        for constraint in constraints
                    ]
                )
            )

        bonds = []
        for i_a, i_b in matching_bonds:
            # if both atoms of the bond are found in a disallowed substructure, we cannot consider them
            # on the other end, a bond between two substructure to preserved independently is perfectly fine
            if any((i_a in ignore_x and i_b in ignore_x) for ignore_x in substructed_ignored):
                continue
            obond = mol.GetBondBetweenAtoms(i_a, i_b)
            bonds.append(obond.GetIdx())

        if len(bonds) > 0:
            mol = Chem.FragmentOnBonds(
                mol,
                bonds,
                dummyLabels=[(i + bond_map_id, i + bond_map_id) for i in range(len(bonds))],
            )
        # here we need to be clever and disable rooted atom as the atom with mapping

        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if should_randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(
                frags,
                key=lambda x: x.GetNumAtoms(),
                reverse=True,
            )

        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [
                atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0
            ]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=True,  # needs to always be true
                    rootedAtAtom=non_map_atom_idxs[0] if non_map_atom_idxs else -1,
                )
            )

        scaffold_str = ".".join(frags_str)
        # EN: fix for https://github.com/datamol-io/safe/issues/37
        # we were using the wrong branch number count which did not take into account
        # possible change in digit utilization after bond slicing
        scf_branch_num = self._find_branch_number(scaffold_str) + branch_numbers

        # don't capture atom mapping in the scaffold
        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str))
        # Set iteration made non-canonical encodings, and therefore seeded
        # model prompts, depend on PYTHONHASHSEED. Retain the historical seed-0
        # ordering explicitly while canonical encodings keep ascending order.
        attach_pos = sorted(attach_pos, reverse=not canonical)
        starting_num = 1 if len(scf_branch_num) == 0 else max(scf_branch_num) + 1
        for attach in attach_pos:
            val = self._format_ring_closure(starting_num)
            # we cannot have anything of the form "\([@=-#-$/\]*\d+\)"
            attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
            # check if we have at least 2 matches, if not, we have a dummy
            n_matches = len(attach_regexp.findall(scaffold_str))
            attachment_match = re.fullmatch(r"\[(\d+)\*\]", attach)
            is_explicit_attachment = (
                attachment_match is not None
                and int(attachment_match.group(1)) in open_attachment_ids
            )
            scaffold_str = (
                attach_regexp.sub(val, scaffold_str)
                if n_matches > 1 or is_explicit_attachment
                else scaffold_str.replace(attach, "*")
            )
            starting_num += 1

        # now we need to remove all the parenthesis around digit only number
        wrong_attach = re.compile(r"(?<!%)\((%\(\d+\)|[\%\d]*)\)")
        scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
        # furthermore, we autoapply rdkit-compatible digit standardization.
        if rdkit_safe:
            pattern = r"\(([=-@#\/\\]{0,2})(%\(\d+\)|%?\d{1,2})\)"
            replacement = r"\g<1>\g<2>"
            scaffold_str = re.sub(pattern, replacement, scaffold_str)
        if not self.ignore_stereo and has_specified_stereo:
            source_graph = self._canonical_isomeric_graph(source_mol)
            encoded_graph = self._canonical_isomeric_graph(
                self.decoder(
                    scaffold_str,
                    canonical=True,
                    remove_dummies=False,
                )
            )
            if source_graph is None or source_graph != encoded_graph:
                # Some constrained stereochemical systems can change their
                # RDKit assignment after fragmentation even when no directly
                # stereogenic bond was cut. Preserve the valid input intact.
                if source_text is not None:
                    return source_text
                return Chem.MolToSmiles(
                    source_mol,
                    canonical=True,
                    isomericSmiles=True,
                )
        return scaffold_str


def encode(
    inp: Union[str, dm.Mol],
    canonical: bool = True,
    randomize: Optional[bool] = False,
    seed: Optional[int] = None,
    slicer: Optional[Union[List[str], str, Callable]] = None,
    require_hs: Optional[bool] = None,
    constraints: Optional[List[dm.Mol]] = None,
    ignore_stereo: Optional[bool] = False,
    allow_empty: bool = False,
):
    """
    Convert input smiles to SAFE representation

    Args:
        inp: input smiles
        canonical: whether to return canonical SAFE string. Defaults to True
        randomize: whether to randomize the safe string encoding. Will be ignored if canonical is provided
        seed: optional seed to use when allowing randomization of the SAFE encoding.
        slicer: slicer algorithm to use for encoding. Defaults to "brics".
        require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added.
        constraints: List of molecules or pattern to preserve during the SAFE construction.
        ignore_stereo: whether to discard input stereochemistry explicitly. When false,
            stereochemistry-changing cuts are skipped and the encoded graph is verified.
        allow_empty: whether to tolerate molecules the slicer cannot cut. When True,
            an input with no breakable bonds (for example a rigid ring, a single atom,
            or the components of a salt) is returned as a single unfragmented SAFE
            block instead of raising ``SAFEFragmentationError``.
    """
    if slicer is None:
        slicer = "brics"
    with dm.without_rdkit_log():
        safe_obj = SAFEConverter(slicer=slicer, require_hs=require_hs, ignore_stereo=ignore_stereo)
        try:
            encoded = safe_obj.encoder(
                inp,
                canonical=canonical,
                randomize=randomize,
                constraints=constraints,
                seed=seed,
                allow_empty=allow_empty,
            )
        except (SAFEEncodeError, SAFEFragmentationError) as e:
            raise e
        except Exception as e:
            raise SAFEEncodeError(f"Failed to encode {inp} with {slicer}") from e
        return encoded


def decode(
    safe_str: str,
    as_mol: bool = False,
    canonical: bool = False,
    fix: bool = True,
    remove_added_hs: bool = True,
    remove_dummies: bool = True,
    ignore_errors: bool = False,
):
    """Convert input SAFE representation to smiles
    Args:
        safe_str: input SAFE representation to decode as a valid molecule or smiles
        as_mol: whether to return a molecule object or a smiles string
        canonical: whether to return a canonical smiles or a randomized smiles
        fix: whether to fix the SAFE representation to take into account non-connected attachment points
        remove_added_hs: whether to remove the hydrogen atoms that have been added to fix the string.
        remove_dummies: whether to remove dummy atoms from the SAFE representation
        ignore_errors: whether to ignore error and return None on decoding failure or raise an error

    """
    with dm.without_rdkit_log():
        safe_obj = SAFEConverter()
        try:
            decoded = safe_obj.decoder(
                safe_str,
                as_mol=as_mol,
                canonical=canonical,
                fix=fix,
                remove_dummies=remove_dummies,
                remove_added_hs=remove_added_hs,
            )

        except Exception as e:
            if ignore_errors:
                return None
            raise SAFEDecodeError(f"Failed to decode {safe_str}") from e
        return decoded
