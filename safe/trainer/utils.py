from typing import Optional
from typing import Any
from typing import List
from typing import Tuple
from functools import partial
from collections import deque
from itertools import combinations
from networkx.utils import py_random_state
from rdkit.Chem import EditableMol, Atom

import networkx as nx
import numpy as np
import random
import datamol as dm
import safe as sf

__implicit_carbon_query = dm.from_smarts("[#6;h]")
__mmpa_query = dm.from_smarts("[*;!$(*=,#[!#6])]!@!=!#[*]")


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
def mol_partition(mol, query: Optional[dm.Mol] = None, seed: Optional[int] = None, **kwargs: Any):
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
    inner_partition = list(sorted(nx.connected_components(subgraphs), key=lambda x: min(x)))
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
    mol,
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
    except sf.SafeFragmentationError:
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
            except (sf.SafeEncodeError, sf.SafeFragmentationError) as e:
                # logger.exception(e)
                return x
        # we need to resplit using attachment point but here we are only adding
    except sf.SafeEncodeError as e:
        return x
    return x
