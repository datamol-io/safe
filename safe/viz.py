import itertools
from typing import Any, Optional, Tuple, Union

import datamol as dm
import matplotlib.pyplot as plt

import safe as sf


def to_image(
    safe_str: str,
    fragments: Optional[Union[str, dm.Mol]] = None,
    legend: Union[str, None] = None,
    mol_size: Union[Tuple[int, int], int] = (300, 300),
    use_svg: Optional[bool] = True,
    highlight_mode: Optional[str] = "lasso",
    highlight_bond_width_multiplier: int = 12,
    **kwargs: Any,
):
    """Display a safe string by highlighting the fragments that make it.

    Args:
        safe_str: the safe string to display
        fragments: list of fragment to highlight on the molecules. If None, will use safe decomposition of the molecule.
        legend: A string to use as the legend under the molecule.
        mol_size: The size of the image to be returned
        use_svg: Whether to return an svg or png image
        highlight_mode: the highlight mode to use. One of ["lasso", "fill", "color"]. If None, no highlight will be shown
        highlight_bond_width_multiplier: the multiplier to use for the bond width when using the 'fill' mode
        **kwargs: Additional arguments to pass to the drawing function. See RDKit
            documentation related to `MolDrawOptions` for more details at
            https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html.

    """

    kwargs["legends"] = legend
    kwargs["mol_size"] = mol_size
    kwargs["use_svg"] = use_svg
    if highlight_bond_width_multiplier is not None:
        kwargs["highlightBondWidthMultiplier"] = highlight_bond_width_multiplier

    if highlight_mode == "color":
        kwargs["continuousHighlight"] = False
        kwargs["circleAtoms"] = kwargs.get("circleAtoms", False) or False

    if isinstance(fragments, (str, dm.Mol)):
        fragments = [fragments]

    if fragments is None and highlight_mode is not None:
        fragments = [
            sf.decode(x, as_mol=False, remove_dummies=False, ignore_errors=False)
            for x in safe_str.split(".")
        ]
    elif fragments and len(fragments) > 0:
        parsed_fragments = []
        for fg in fragments:
            if isinstance(fg, str) and dm.to_mol(fg) is None:
                fg = sf.decode(fg, as_mol=False, remove_dummies=False, ignore_errors=False)
            parsed_fragments.append(fg)
        fragments = parsed_fragments
    else:
        fragments = []
    mol = dm.to_mol(safe_str, remove_hs=False)
    cm = plt.get_cmap("gist_rainbow")
    current_colors = [cm(1.0 * i / len(fragments)) for i in range(len(fragments))]

    if highlight_mode == "lasso":
        return dm.viz.lasso_highlight_image(mol, fragments, **kwargs)

    atom_indices = []
    bond_indices = []
    atom_colors = {}
    bond_colors = {}

    for i, frag in enumerate(fragments):
        frag = dm.from_smarts(frag)
        atom_matches, bond_matches = dm.substructure_matching_bonds(mol, frag)
        atom_matches = list(itertools.chain(*atom_matches))
        bond_matches = list(itertools.chain(*bond_matches))
        atom_indices.extend(atom_matches)
        bond_indices.extend(bond_matches)
        atom_colors.update({x: current_colors[i] for x in atom_matches})
        bond_colors.update({x: current_colors[i] for x in bond_matches})

    return dm.viz.to_image(
        mol,
        highlight_atom=[atom_indices],
        highlight_bond=[bond_indices],
        highlightAtomColors=[atom_colors],
        highlightBondColors=[bond_colors],
        **kwargs,
    )
