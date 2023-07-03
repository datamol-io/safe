from typing import Union
from typing import Optional
from typing import List
from typing import Any
from typing import Callable

import re
import datamol as dm
import itertools
import numpy as np

from collections import Counter
from loguru import logger

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import MolToSmiles


class SafeConverter:
    """Molecule line notation conversion from SMILES to SAFE

    A SAFE representation is a string based representation of a molecule decomposition into fragment components, separated by a dot ('.').
    Note that each component (fragment) might not be a valid molecule by themselves, unless explicitely correct to add missing hydrogens.
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
        "mmpa": ["[#6+0;!$(*=, #[!#6])]!@!=!#[*]"],  # classical mmpa slicing smarts
        "attach": ["[*]!@[*]"],  # any potential attachment point, including hydrogens when explicit
    }

    def randomize(self, mol: dm.Mol, rng: Optional[int] = None):
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

    def __init__(
        self,
        fragmentation: Optional[Union[str, List[str], Callable]] = "brics",
        require_hs: Optional[bool] = None,
    ):
        """Constructor for the SAFE converter

        Args:
            fragmentation:  fragmentation algorithm to use for encoding. Can either be one of the supported slicing algorithm (SUPPORTED_SLICERS)
                or a custom callable that returns the bond ids that can be sliced.
            require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added. `attach` fragmentation requires adding hydrogens.

        """
        self.fragmentation = fragmentation
        if isinstance(fragmentation, str) and fragmentation.lower() in self.SUPPORTED_SLICERS:
            self.fragmentation = self.__SLICE_SMARTS.get(fragmentation.lower(), fragmentation)
        if isinstance(self.fragmentation, (list, tuple)):
            self.fragmentation = [dm.from_smarts(x) for x in self.fragmentation]
        self.require_hs = require_hs or (fragmentation == "attach")

    def decoder(
        self,
        inp: str,
        return_mol: bool = False,
        canonical: bool = False,
        fix: bool = True,
        remove_dummies: bool = True,
    ):
        """Convert input scaffold-core representation to smiles

        Args:
            inp: input scaffold-core representation to decode as a valid molecules or smiles
            return_mol: whether to return a molecule object or a smiles string
            canonical: whether to return a canonical smiles or a randomized smiles
            fix: whether to fix the scaffold-core representation to take into account non-connected attachment points
            remove_dummies: whether to remove dummy atoms from the scaffold-core representation

        """
        missing_tokens = [inp]
        if fix:
            branch_numbers = [
                int(x.replace("%", "")) for x in re.findall(r"[^\[](((?<=%)\d{2})|(\d))[^\]]", inp)
            ]
            # any branch number that is not pairwise should receive a dummy atom to complete the attachment point
            branch_numbers = Counter(branch_numbers)
            for i, (bnum, bcount) in enumerate(branch_numbers.items()):
                if bcount % 2 != 0:
                    bnum_str = str(bnum) if bnum < 10 else f"%{bnum}"
                    missing_tokens.append(f"[*:{i+1}]{bnum_str}")

        mol = dm.to_mol(".".join(missing_tokens))
        if remove_dummies:
            mol = dm.remove_dummies(mol)
        if return_mol:
            return mol
        return dm.to_smiles(mol, canonical=canonical)

    def _fragment(self, mol: dm.Mol):
        """
        Perform bond cutting in place for the input molecule, given the slicing algorithm

        Args:
            mol: input molecule to split
        """

        if self.require_hs:
            mol = dm.add_hs(mol)

        if callable(self.fragmentation):
            matching_bonds = self.fragmentation(mol)

        elif self.fragmentation == "brics":
            matching_bonds = BRICS.FindBRICSBonds(mol)
            matching_bonds = [brics_match[0] for brics_match in matching_bonds]

        else:
            matches = set()
            for smarts in self.fragmentation:
                matches |= set(tuple(sorted(match)) for match in mol.GetSubstructMatches(smarts))
            matching_bonds = list(matching_bonds)

        return matching_bonds

    def encoder(
        self,
        inp: Union[str, dm.Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[dm.Mol]] = None,
    ):
        """Convert input smiles to scaffold-core representation

        Args:
            inp: input smiles
            bond_smarts: decomposition type or smarts string or list of smarts strings to use for bond breaking
            canonical: whether to return canonical smiles string. Defaults to True
            randomize: whether to randomize the safe string encoding. Will automatically set canonical to False for smiles generation
            seed: optional seed to use when allowing randomization of the SAFE encoding. Randomization happens at two steps:
                1. at the original smiles representation by randomization the atoms.
                2. at the SAFE conversion by randomizing fragment orders
            constraints: List of molecules or pattern to preserve during the SAFE construction.
                Any bond slicing would happen inside a substructure matching any of the patterns would not be taken into account

        """
        rng = None
        if randomize:
            rng = np.random.default_rng(seed)
            inp = dm.to_mol(inp)
            inp = self.randomize(inp, rng)

        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(inp, canonical=(not randomize), randomize=False, ordered=False)

        branch_numbers = [int(x) for x in re.findall(r"[^\[]%?(\d+)[^\]]", inp)]
        mol = dm.to_mol(inp)
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

        if canonical:
            frags = list(
                sorted(
                    frags,
                    key=lambda x: x.GetNumAtoms(),
                    reverse=True,
                )
            )
        elif randomize:
            frags = rng.permutation(frags).tolist()
        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() != "*"]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=canonical,
                    rootedAtAtom=non_map_atom_idxs[0],
                )
            )

        scaffold_str = ".".join(frags_str)
        attach_pos = set(re.findall(r"(\[\d+\*\]|\[[^:]*:\d+\])", scaffold_str))
        starting_num = max(branch_numbers) + 1

        for attach in attach_pos:
            val = str(starting_num) if starting_num < 10 else f"%{starting_num}"
            scaffold_str = scaffold_str.replace("(" + attach + ")", val).replace(attach, val)
            starting_num += 1
        return scaffold_str


def encode(inp: Union[str, dm.Mol], canonical: bool = True):
    pass
