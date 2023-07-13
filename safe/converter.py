from typing import Union
from typing import Optional
from typing import List
from typing import Callable

import re
import datamol as dm
import itertools
import numpy as np

from collections import Counter

from rdkit import Chem
from rdkit.Chem import BRICS
from ._exception import SafeDecodeError
from ._exception import SafeEncodeError
from ._exception import SafeFragmentationError


class SafeConverter:
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

    SUPPORTED_SLICERS = ["hr", "recap", "mmpa", "attach", "brics"]
    __SLICE_SMARTS = {
        "hr": ["[*]!@-[*]"],  # any non ring single bond
        "recap": [
            "[C;$(C=O)]!@-N",  # amides and urea
            "[C;$(C=O)]!@-O",  # esters
            "C!@-[N;!$(NC=O)]",  # amines
            "C!@-[O;!$(NC=O)]",  # ether
            "[CX3]!@=[CX3]",  # olefin
            "[N+X4]!@-C",  # quaternary nitrogen
            "n!@-C",  # aromatic N - aliphatic C
            "[$([NR][CR]=O)]!@-C",  # lactam nitrogen - aliphatic carbon
            "c!@-c",  # aromatic C - aromatic C
            "N!@-[$(S(=O)=O)]",  # sulphonamides
        ],
        "mmpa": ["[#6+0;!$(*=,#[!#6])]!@!=!#[*]"],  # classical mmpa slicing smarts
        "attach": ["[*]!@[*]"],  # any potential attachment point, including hydrogens when explicit
    }

    def __init__(
        self,
        slicer: Optional[Union[str, List[str], Callable]] = "brics",
        require_hs: Optional[bool] = None,
    ):
        """Constructor for the SAFE converter

        Args:
            slicer: slicer algorithm to use for encoding.
                Can either be one of the supported slicing algorithm (SUPPORTED_SLICERS)
                or a custom callable that returns the bond ids that can be sliced.
            require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added.
                `attach` slicer requires adding hydrogens.

        """
        self.slicer = slicer
        if isinstance(slicer, str) and slicer.lower() in self.SUPPORTED_SLICERS:
            self.slicer = self.__SLICE_SMARTS.get(slicer.lower(), slicer)
        if self.slicer != "brics" and isinstance(self.slicer, str):
            self.slicer = [self.slicer]
        if isinstance(self.slicer, (list, tuple)):
            self.slicer = [dm.from_smarts(x) for x in self.slicer]
        self.require_hs = require_hs or (slicer == "attach")

    @staticmethod
    def randomize(mol: dm.Mol, rng: Optional[int] = None):
        """Randomize the position of the atoms in a mol.

        Args:
            mol: molecules to randomize
            seed: optional seed to use
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        if mol.GetNumAtoms() == 0:
            return mol
        atom_indices = list(range(mol.GetNumAtoms()))
        atom_indices = rng.permutation(atom_indices).tolist()
        return Chem.RenumberAtoms(mol, atom_indices)

    def _find_branch_number(self, inp: str):
        """Find the branch number and ring closure in the SMILES representation using regexp

        Args:
            inp: input smiles
        """

        matching_groups = re.findall(r"((?<=%)\d{2})|((?<!%)\d+)", inp)
        # first match is for multiple connection as multiple digits
        # second match is for single connections requiring 2 digits
        # SMILES does not support triple digits
        branch_numbers = []
        for m in matching_groups:
            if m[0] == "":
                branch_numbers.extend(int(mm) for mm in m[1])
            elif m[1] == "":
                branch_numbers.append(int(m[0].replace("%", "")))
        return branch_numbers

    def decoder(
        self,
        inp: str,
        as_mol: bool = False,
        canonical: bool = False,
        fix: bool = True,
        remove_dummies: bool = True,
    ):
        """Convert input SAFE representation to smiles

        Args:
            inp: input SAFE representation to decode as a valid molecule or smiles
            as_mol: whether to return a molecule object or a smiles string
            canonical: whether to return a canonical smiles or a randomized smiles
            fix: whether to fix the SAFE representation to take into account non-connected attachment points
            remove_dummies: whether to remove dummy atoms from the SAFE representation

        """
        missing_tokens = [inp]
        if fix:
            branch_numbers = self._find_branch_number(inp)
            # only use the set that have exactly 1 element
            # any branch number that is not pairwise should receive a dummy atom to complete the attachment point
            branch_numbers = Counter(branch_numbers)
            for i, (bnum, bcount) in enumerate(branch_numbers.items()):
                if bcount % 2 != 0:
                    bnum_str = str(bnum) if bnum < 10 else f"%{bnum}"
                    missing_tokens.append(f"[*:{i+1}]{bnum_str}")

        mol = dm.to_mol(".".join(missing_tokens))
        if remove_dummies:
            mol = dm.remove_dummies(mol)
        if as_mol:
            return mol
        return dm.to_smiles(mol, canonical=canonical, explicit_hs=True)

    def _fragment(self, mol: dm.Mol):
        """
        Perform bond cutting in place for the input molecule, given the slicing algorithm

        Args:
            mol: input molecule to split
        Raises:
            SafeFragmentationError: if the slicing algorithm return empty bonds
        """

        if callable(self.slicer):
            matching_bonds = self.slicer(mol)

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
        if matching_bonds is None or len(matching_bonds) == 0:
            raise SafeFragmentationError(
                "Slicing algorithms did not return any bonds that can be cut !"
            )
        return matching_bonds

    def encoder(
        self,
        inp: Union[str, dm.Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[dm.Mol]] = None,
    ):
        """Convert input smiles to SAFE representation

        Args:
            inp: input smiles
            bond_smarts: decomposition type or smarts string or list of smarts strings to use for bond breaking
            canonical: whether to return canonical smiles string. Defaults to True
            randomize: whether to randomize the safe string encoding. Will be ignored if canonical is provided
            seed: optional seed to use when allowing randomization of the SAFE encoding.
                Randomization happens at two steps:
                1. at the original smiles representation by randomization the atoms.
                2. at the SAFE conversion by randomizing fragment orders
            constraints: List of molecules or pattern to preserve during the SAFE construction. Any bond slicing would
                happen outside of a substructure matching one of the patterns.
        """
        rng = None
        if randomize and not canonical:
            rng = np.random.default_rng(seed)
            inp = dm.to_mol(inp, remove_hs=False)
            inp = self.randomize(inp, rng)

        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(inp, canonical=canonical, randomize=False, ordered=False)

        # TODO(maclandrol): RDKit supports some extended form of ring closure, up to 5 digits
        # https://www.rdkit.org/docs/RDKit_Book.html#ring-closures and I should try to include them
        branch_numbers = self._find_branch_number(inp)
        mol = dm.to_mol(inp, remove_hs=False)
        if self.require_hs:
            mol = dm.add_hs(mol)
        matching_bonds = self._fragment(mol)
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
        mol = Chem.FragmentOnBonds(
            mol, bonds, dummyLabels=[(i + 1, i + 1) for i in range(len(bonds))]
        )
        # here we need to be clever and disable rooted atom as the atom with mapping
        frags = list(Chem.GetMolFrags(mol, asMols=True))

        if randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(
                frags,
                key=lambda x: x.GetNumAtoms(),
                reverse=True,
            )
        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() != "*"]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=True,  # needs to always be true
                    rootedAtAtom=non_map_atom_idxs[0],
                )
            )

        scaffold_str = ".".join(frags_str)
        attach_pos = set(re.findall(r"(\[\d+\*\]|\[[^:]*:\d+\])", scaffold_str))
        if len(branch_numbers) == 0:
            starting_num = 1
        else:
            starting_num = max(branch_numbers) + 1
        for attach in attach_pos:
            val = str(starting_num) if starting_num < 10 else f"%{starting_num}"
            # we cannot have anything of the form "\([@=-#-$/\]*\d+\)"
            attach_regexp = re.compile(r"\(?([\.:@=\-#$/\\])?(" + re.escape(attach) + r")\)?")
            scaffold_str = attach_regexp.sub(r"\g<1>" + val, scaffold_str)
            starting_num += 1
        return scaffold_str


def encode(
    inp: Union[str, dm.Mol],
    canonical: bool = True,
    randomize: Optional[bool] = False,
    seed: Optional[int] = None,
    slicer: Optional[Union[str, Callable]] = None,
    require_hs: Optional[bool] = None,
    constraints: Optional[List[dm.Mol]] = None,
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
    """
    if slicer is None:
        slicer = "brics"
    with dm.without_rdkit_log():
        safe = SafeConverter(slicer=slicer, require_hs=require_hs)
        try:
            encoded = safe.encoder(
                inp, canonical=canonical, randomize=randomize, constraints=constraints, seed=seed
            )
        except SafeFragmentationError as e:
            raise e
        except Exception as e:
            raise SafeEncodeError(f"Failed to encode {inp} with {slicer}") from e
        return encoded


def decode(
    safe_str: str,
    as_mol: bool = False,
    canonical: bool = False,
    fix: bool = True,
    remove_dummies: bool = True,
):
    """Convert input SAFE representation to smiles
    Args:
        inp: input SAFE representation to decode as a valid molecule or smiles
        as_mol: whether to return a molecule object or a smiles string
        canonical: whether to return a canonical smiles or a randomized smiles
        fix: whether to fix the SAFE representation to take into account non-connected attachment points
        remove_dummies: whether to remove dummy atoms from the SAFE representation

    """
    with dm.without_rdkit_log():
        safe = SafeConverter()
        try:
            decoded = safe.decoder(
                safe_str, as_mol=as_mol, canonical=canonical, fix=fix, remove_dummies=remove_dummies
            )
        except Exception as e:
            raise SafeDecodeError(f"Failed to decode {safe_str}") from e
        return decoded
