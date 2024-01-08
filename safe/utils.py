import itertools
import random
import re
from collections import deque
from contextlib import contextmanager, suppress
from functools import partial
from itertools import combinations, compress
from typing import Any, List, Optional, Tuple, Union

import datamol as dm
import networkx as nx
import numpy as np
from loguru import logger
from networkx.utils import py_random_state
from rdkit import Chem
from rdkit.Chem import Atom, EditableMol
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem.rdmolops import AdjustQueryParameters, AdjustQueryProperties, ReplaceCore

import safe as sf

__implicit_carbon_query = dm.from_smarts("[#6;h]")
__mmpa_query = dm.from_smarts("[*;!$(*=,#[!#6])]!@!=!#[*]")

_SMILES_ATTACHMENT_POINT_TOKEN = "\\*"
# The following allows discovering and extracting valid dummy atoms in smiles/smarts.
_SMILES_ATTACHMENT_POINTS = [
    # parse any * not preceeded by "[" or ":"" and not followed by "]" or ":" as attachment
    r"(?<![:\[]){0}(?![:\]])".format(_SMILES_ATTACHMENT_POINT_TOKEN),
    # fix some edge cases of the above for attachment in the "()" environment or in SMARTS notation
    r"(?<![\[(\[\w)]){0}".format(_SMILES_ATTACHMENT_POINT_TOKEN),
    # parse attachment in optionally mapped isotope smarts atoms. eg. [1*:3]
    r"\[(\d*){0}:?(\d*)\]".format(_SMILES_ATTACHMENT_POINT_TOKEN),
]


class MolSlicer:
    """Slice a molecule into head-linker-tail"""

    BOND_SPLITTERS = [
        # two atoms connected by a non ring single bond, one of each is not in a ring and at least two heavy neighbor
        "[R:1]-&!@[!R;!D1:2]",
        # two atoms in different rings linked by a non-ring single bond
        "[R:1]-&!@[R:2]",
    ]
    _BOND_BUFFER = 1  # buffer around substructure match size.
    MAX_CUTS = 2  # maximum number of cuts. Here we need two cuts for head-linker-tail.

    _MERGING_RXN = dm.reactions.rxn_from_smarts(
        "[#0][*:1].[#0][*:4].([#0][*:2].[#0][*:3])>>([*:1][*:2].[*:3][*:4])"
    )

    def __init__(
        self,
        shortest_linker: bool = False,
        min_linker_size: int = 0,
        require_ring_system: bool = True,
        verbose: bool = False,
    ):
        """
        Constructor of bond slicer.

        Args:
            shortest_linker: whether to consider longuest or shortest linker.
                Does not have any effect when expected_head group is provided during splitting
            min_linker_size: minimum linker size
            require_ring_system: whether all fragment needs to have a ring system
            verbose: whether to allow verbosity in logging
        """

        self.bond_splitters = [dm.from_smarts(x) for x in self.BOND_SPLITTERS]
        self.shortest_linker = shortest_linker
        self.min_linker_size = min_linker_size
        self.require_ring_system = require_ring_system
        self.verbose = verbose

    def get_ring_system(self, mol: dm.Mol):
        """Get the list of ring system from a molecule

        Args:
            mol: input molecule for which we are computing the ring system
        """
        mol.UpdatePropertyCache()
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ring_atoms = set(ring)
            cur_system = []  # keep a track of ring system
            for system in systems:
                if len(ring_atoms.intersection(system)) > 0:
                    ring_atoms = ring_atoms.union(system)  # merge ring system that overlap
                else:
                    cur_system.append(system)
            cur_system.append(ring_atoms)
            systems = cur_system
        return systems

    def _bond_selection_from_max_cuts(self, bond_list: List[int], dist_mat: np.ndarray):
        """Select bonds based on maximum number of cuts allowed"""
        # for now we are just implementing to 2 max cuts algorithms
        if self.MAX_CUTS != 2:
            raise ValueError(f"Only MAX_CUTS=2 is supported, got {self.MAX_CUTS}")

        bond_pdist = np.full((len(bond_list), len(bond_list)), -1)
        for i in range(len(bond_list)):
            for j in range(i, len(bond_list)):
                # we get the minimum topological distance between bond to cut
                bond_pdist[i, j] = bond_pdist[j, i] = min(
                    [dist_mat[a1, a2] for a1, a2 in itertools.product(bond_list[i], bond_list[j])]
                )

        masked_bond_pdist = np.ma.masked_less_equal(bond_pdist, self.min_linker_size)

        if self.shortest_linker:
            return np.unravel_index(np.ma.argmin(masked_bond_pdist), bond_pdist.shape)
        return np.unravel_index(np.ma.argmax(masked_bond_pdist), bond_pdist.shape)

    def _get_bonds_to_cut(self, mol: dm.Mol):
        """Get possible bond to cuts

        Args:
            mol: input molecule
        """
        # use this if you want to enumerate yourself the possible cuts

        ring_systems = self.get_ring_system(mol)
        candidate_bonds = []
        ring_query = Chem.rdqueries.IsInRingQueryAtom()

        for query in self.bond_splitters:
            bonds = mol.GetSubstructMatches(query, uniquify=True)
            cur_unique_bonds = [set(cbond) for cbond in candidate_bonds]
            # do not accept bonds part of the same ring system or already known
            for b in bonds:
                bond_id = mol.GetBondBetweenAtoms(*b).GetIdx()
                bond_cut = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(mol, [bond_id], addDummies=False), asMols=True
                )
                can_add = not self.require_ring_system or all(
                    len(frag.GetAtomsMatchingQuery(ring_query)) > 0 for frag in bond_cut
                )
                if can_add and not (
                    set(b) in cur_unique_bonds or any(x.issuperset(set(b)) for x in ring_systems)
                ):
                    candidate_bonds.append(b)
        return candidate_bonds

    def _fragment_mol(self, mol: dm.Mol, bonds: List[dm.Bond]):
        """Fragment molecules on bonds and return head, linker, tail combination

        Args:
            mol: input molecule
            bonds: list of bonds to cut
        """
        tmp = Chem.rdmolops.FragmentOnBonds(mol, [b.GetIdx() for b in bonds])
        _frags = list(Chem.GetMolFrags(tmp, asMols=True))
        # linker is the one with 2 dummy atoms
        linker_pos = 0
        for pos, _frag in enumerate(_frags):
            if sum([at.GetSymbol() == "*" for at in _frag.GetAtoms()]) == 2:
                linker_pos = pos
                break
        linker = _frags.pop(linker_pos)
        head, tail = _frags
        return (head, linker, tail)

    def _compute_linker_score(self, linker: dm.Mol):
        """Compute the score of a linker to help select between linkers"""

        # we need to take into account
        # case where we require the linker to have a ring system
        # case where we want the linker to be longuest or shortest

        # find shortest path
        attach1, attach2, *_ = [at.GetIdx() for at in linker.GetAtoms() if at.GetSymbol() == "*"]
        score = len(Chem.rdmolops.GetShortestPath(linker, attach1, attach2))
        ring_query = Chem.rdqueries.IsInRingQueryAtom()
        linker_ring_count = len(linker.GetAtomsMatchingQuery(ring_query))
        if self.require_ring_system:
            score *= int(linker_ring_count > 0)
        if score == 0:
            return float("inf")
        if not self.shortest_linker:
            score = 1 / score
        return score

    def __call__(self, mol: Union[dm.Mol, str], expected_head: Union[dm.Mol, str] = None):
        """Perform slicing of the input molecule

        Args:
            mol: input molecule
            expected_head: substructure that should be part of the head.
                The small fragment containing this substructure would be kept as head
        """

        mol = dm.to_mol(mol)
        # remove salt and solution
        mol = dm.keep_largest_fragment(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
        dist_mat = Chem.rdmolops.GetDistanceMatrix(mol)

        if expected_head is not None:
            if isinstance(expected_head, str):
                expected_head = dm.to_mol(expected_head)
            if not mol.HasSubstructMatch(expected_head):
                if self.verbose:
                    logger.info(
                        "Expected head was provided, but does not match molecules. It will be ignored"
                    )
                expected_head = None

        candidate_bonds = self._get_bonds_to_cut(mol)

        # we have all the candidate bonds we can cut
        # now we need to pick the most plausible bonds
        selected_bonds = [mol.GetBondBetweenAtoms(a1, a2) for (a1, a2) in candidate_bonds]

        # CASE 1: no bond to cut ==> only head
        if len(selected_bonds) == 0:
            return (mol, None, None)

        # CASE 2: only one bond ==> linker is empty
        if len(selected_bonds) == 1:
            # there is not linker
            tmp = Chem.rdmolops.FragmentOnBonds(mol, [b.GetIdx() for b in selected_bonds])
            head, tail = Chem.GetMolFrags(tmp, asMols=True)
            return (head, None, tail)

        # CASE 3a: we select the most plausible bond to cut on ourselves
        if expected_head is None:
            choice = self._bond_selection_from_max_cuts(candidate_bonds, dist_mat)
            selected_bonds = [selected_bonds[c] for c in choice]
            return self._fragment_mol(mol, selected_bonds)

        # CASE 3b: slightly more complex case where we want the head to be the smallest graph containing the
        # provided substructure
        bond_combination = list(itertools.combinations(selected_bonds, self.MAX_CUTS))
        bond_score = float("inf")
        linker_score = float("inf")
        head, linker, tail = (None, None, None)
        for split_bonds in bond_combination:
            cur_head, cur_linker, cur_tail = self._fragment_mol(mol, split_bonds)
            # head can also be tail
            head_match = cur_head.GetSubstructMatch(expected_head)
            tail_match = cur_tail.GetSubstructMatch(expected_head)
            if not head_match and not tail_match:
                continue
            if not head_match and tail_match:
                cur_head, cur_tail = cur_tail, cur_head
            cur_bond_score = cur_head.GetNumHeavyAtoms()
            # compute linker score
            cur_linker_score = self._compute_linker_score(cur_linker)
            if (cur_bond_score < bond_score) or (
                cur_bond_score < self._BOND_BUFFER + bond_score and cur_linker_score < linker_score
            ):
                head, linker, tail = cur_head, cur_linker, cur_tail
                bond_score = cur_bond_score
                linker_score = cur_linker_score

        return (head, linker, tail)

    @classmethod
    def link_fragments(
        cls, linker: Union[dm.Mol, str], head: Union[dm.Mol, str], tail: Union[dm.Mol, str]
    ):
        """Link fragments together using the provided linker

        Args:
            linker: linker to use
            head: head fragment
            tail: tail fragment
        """
        if isinstance(linker, dm.Mol):
            linker = dm.to_smiles(linker)
        linker = standardize_attach(linker)
        reactants = [dm.to_mol(head), dm.to_mol(tail), dm.to_mol(linker)]
        return dm.reactions.apply_reaction(
            cls._MERGING_RXN, reactants, as_smiles=True, sanitize=True, product_index=0
        )


@contextmanager
def attr_as(obj: Any, field: str, value: Any):
    """Temporary replace the value of an object

    Args:
        obj: object to temporary patch
        field: name of the key to change
        value: value of key to be temporary changed
    """
    old_value = getattr(obj, field, None)
    setattr(obj, field, value)
    yield
    with suppress(TypeError):
        setattr(obj, field, old_value)


def _selective_add_hs(mol: dm.Mol, fraction_hs: Optional[bool] = None):
    """Custom addition of hydrogens to a molecule
    This version of hydrogen bond adding only at max 1 hydrogen per atom

    Args:
        mol: molecule to split
        fraction_hs: proportion of random atom to which we will add explicit hydrogens
    """

    carbon_with_implicit_atoms = mol.GetSubstructMatches(__implicit_carbon_query, uniquify=True)
    carbon_with_implicit_atoms = [x[0] for x in carbon_with_implicit_atoms]
    carbon_with_implicit_atoms = list(set(carbon_with_implicit_atoms))
    # we get a proportion of the carbon we can extend
    if fraction_hs is not None and fraction_hs > 0:
        fraction_hs = np.ceil(fraction_hs * len(carbon_with_implicit_atoms))
        fraction_hs = int(np.clip(fraction_hs, 1, len(carbon_with_implicit_atoms)))
        carbon_with_implicit_atoms = random.sample(carbon_with_implicit_atoms, k=fraction_hs)
    carbon_with_implicit_atoms = [int(x) for x in carbon_with_implicit_atoms]
    emol = EditableMol(mol)
    for atom_id in carbon_with_implicit_atoms:
        h_atom = emol.AddAtom(Atom("H"))
        emol.AddBond(atom_id, h_atom, dm.SINGLE_BOND)
    return emol.GetMol()


@py_random_state("seed")
def mol_partition(
    mol: dm.Mol, query: Optional[dm.Mol] = None, seed: Optional[int] = None, **kwargs: Any
):
    """Partition a molecule into fragments using a bond query

    Args:
        mol: molecule to split
        query: bond query to use for splitting
        seed: random seed
        kwargs: additional arguments to pass to the partitioning algorithm

    """
    resolution = kwargs.get("resolution", 1.0)
    threshold = kwargs.get("threshold", 1e-7)
    weight = kwargs.get("weight", "weight")

    if query is None:
        query = __mmpa_query

    G = dm.graph.to_graph(mol)
    bond_partition = [
        tuple(sorted(match)) for match in mol.GetSubstructMatches(query, uniquify=True)
    ]

    def get_relevant_edges(e1, e2):
        return tuple(sorted([e1, e2])) not in bond_partition

    subgraphs = nx.subgraph_view(G, filter_edge=get_relevant_edges)

    partition = [{u} for u in G.nodes()]
    inner_partition = sorted(nx.connected_components(subgraphs), key=lambda x: min(x))
    mod = nx.algorithms.community.modularity(
        G, inner_partition, resolution=resolution, weight=weight
    )
    is_directed = G.is_directed()
    graph = G.__class__()
    graph.add_nodes_from(G)
    graph.add_weighted_edges_from(G.edges(data=weight, default=1))
    graph = nx.algorithms.community.louvain._gen_graph(graph, inner_partition)
    m = graph.size(weight="weight")
    partition, inner_partition, improvement = nx.algorithms.community.louvain._one_level(
        graph, m, inner_partition, resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = nx.algorithms.community.modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = nx.algorithms.community.louvain._gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = nx.algorithms.community.louvain._one_level(
            graph, m, partition, resolution, is_directed, seed
        )


def find_partition_edges(G: nx.Graph, partition: List[List]) -> List[Tuple]:
    """
    Find the edges connecting the subgraphs in a given partition of a graph.

    Args:
        G (networkx.Graph): The original graph.
        partition (list of list of nodes): The partition of the graph where each element is a list of nodes representing a subgraph.

    Returns:
        list: A list of edges connecting the subgraphs in the partition.
    """
    partition_edges = []
    for subgraph1, subgraph2 in combinations(partition, 2):
        edges = nx.edge_boundary(G, subgraph1, subgraph2)
        partition_edges.extend(edges)
    return partition_edges


def fragment_aware_spliting(mol: dm.Mol, fraction_hs: Optional[bool] = None, **kwargs: Any):
    """Custom splitting algorithm for dataset building.

    This slicing strategy will cut any bond including bonding with hydrogens
    However, only one cut per atom is allowed

    Args:
        mol: molecule to split
        fraction_hs: proportion of random atom to which we will add explicit hydrogens
        kwargs: additional arguments to pass to the partitioning algorithm
    """
    random.seed(kwargs.get("seed", 1))
    mol = dm.to_mol(mol, remove_hs=False)
    mol = _selective_add_hs(mol, fraction_hs=fraction_hs)
    graph = dm.graph.to_graph(mol)
    d = mol_partition(mol, **kwargs)
    q = deque(d)
    partition = q.pop()
    return find_partition_edges(graph, partition)


def convert_to_safe(
    mol: dm.Mol,
    canonical: bool = False,
    randomize: bool = False,
    seed: Optional[int] = 1,
    slicer: str = "brics",
    split_fragment: bool = True,
    fraction_hs: bool = None,
    resolution: Optional[float] = 0.5,
):
    """Convert a molecule to a safe representation

    Args:
        mol: molecule to convert
        canonical: whether to use canonical encoding
        randomize: whether to randomize the encoding
        seed: random seed
        slicer: the slicer to use for fragmentation
        split_fragment: whether to split fragments
        fraction_hs: proportion of random atom to which we will add explicit hydrogens
        resolution: resolution for the partitioning algorithm
        seed: random seed
    """
    x = None
    try:
        x = sf.encode(mol, canonical=canonical, randomize=randomize, slicer=slicer, seed=seed)
    except sf.SAFEFragmentationError:
        if split_fragment:
            if "." in mol:
                return None
            try:
                x = sf.encode(
                    mol,
                    canonical=False,
                    randomize=randomize,
                    seed=seed,
                    slicer=partial(
                        fragment_aware_spliting,
                        fraction_hs=fraction_hs,
                        resolution=resolution,
                        seed=seed,
                    ),
                )
            except (sf.SAFEEncodeError, sf.SAFEFragmentationError):
                # logger.exception(e)
                return x
        # we need to resplit using attachment point but here we are only adding
    except sf.SAFEEncodeError:
        return x
    return x


def compute_side_chains(mol: dm.Mol, core: dm.Mol, label_by_index: bool = False):
    """Compute the side chain of a molecule given a core

    !!! note "Finding the side chains"
        The algorithm to find the side chains from core assumes that the core we get as input has attachment points.
        Those attachment points are never considered as part of the query, rather they are used to define the attachment points
        on the side chains. Removing the attachment points from the core is exactly the same as keeping them.

        ```python
        mol = "CC1=C(C(=NO1)C2=CC=CC=C2Cl)C(=O)NC3C4N(C3=O)C(C(S4)(C)C)C(=O)O"
        core0 = "CC1(C)CN2C(CC2=O)S1"
        core1 = "CC1(C)SC2C(-*)C(=O)N2C1-*"
        core2 = "CC1N2C(SC1(C)C)C(N)C2=O"
        side_chain = compute_side_chain(core=core0, mol=mol)
        dm.to_image([side_chain, core0, mol])
        ```
        Therefore on the above, core0 and core1 are equivalent for the molecule `mol`, but core2 is not.

    Args:
        mol: molecule to split
        core: core to use for deriving the side chains
    """

    if isinstance(mol, str):
        mol = dm.to_mol(mol)
    if isinstance(core, str):
        core = dm.to_mol(core)
    core_query_param = AdjustQueryParameters()
    core_query_param.makeDummiesQueries = True
    core_query_param.adjustDegree = False
    core_query_param.aromatizeIfPossible = True
    core_query_param.makeBondsGeneric = False
    core_query = AdjustQueryProperties(core, core_query_param)
    return ReplaceCore(
        mol, core_query, labelByIndex=label_by_index, replaceDummies=False, requireDummyMatch=False
    )


def list_individual_attach_points(mol: dm.Mol, depth: Optional[int] = None):
    """List all individual attachement points.

    We do not allow multiple attachment points per substitution position.

    Args:
        mol: molecule for which we need to open the attachment points

    """
    ATTACHING_RXN = ReactionFromSmarts("[*;h;!$([*][#0]):1]>>[*:1][*]")
    mols = [mol]
    curated_prods = set()
    num_attachs = len(mol.GetSubstructMatches(dm.from_smarts("[*;h:1]"), uniquify=True))
    depth = depth or 1
    depth = min(max(depth, 1), num_attachs)
    while depth > 0:
        prods = set()
        for mol in mols:
            mol = dm.to_mol(mol)
            for p in ATTACHING_RXN.RunReactants((mol,)):
                try:
                    m = dm.sanitize_mol(p[0])
                    sm = dm.to_smiles(m, canonical=True)
                    sm = dm.reactions.add_brackets_to_attachment_points(sm)
                    prods.add(dm.reactions.convert_attach_to_isotope(sm, as_smiles=True))
                except Exception as e:
                    logger.error(e)
        curated_prods.update(prods)
        mols = prods
        depth -= 1
    return list(curated_prods)


def filter_by_substructure_constraints(
    sequences: List[Union[str, dm.Mol]], substruct: Union[str, dm.Mol], n_jobs: int = -1
):
    """Check whether the input substructures are present in each of the molecule in the sequences

    Args:
        sequences: list of molecules to validate
        substruct: substructure to use as query
        n_jobs: number of jobs to use for parallelization

    """

    if isinstance(substruct, str):
        substruct = standardize_attach(substruct)
        substruct = dm.from_smarts(substruct)

    def _check_match(mol):
        with suppress(Exception):
            mol = dm.to_mol(mol)
            return mol.HasSubstructMatch(substruct)
        return False

    matches = dm.parallelized(_check_match, sequences, n_jobs=n_jobs)
    return list(compress(sequences, matches))


def standardize_attach(inputs: str, standard_attach: str = "[*]"):
    """Standardize the attachment points of a molecule

    Args:
        inputs: input molecule
        standard_attach: standard attachment point to use
    """

    for attach_regex in _SMILES_ATTACHMENT_POINTS:
        inputs = re.sub(attach_regex, standard_attach, inputs)
    return inputs
