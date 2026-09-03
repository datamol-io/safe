"""Exercise the installed distribution without importing the checkout."""

import os
from importlib.metadata import distribution
from pathlib import Path

import safe as package
import datamol as dm

installed = distribution("safe-mol")
location = Path(package.__file__).resolve()
assert location == Path(installed.locate_file("safe/__init__.py")).resolve(), location
assert not location.is_relative_to(Path(__file__).resolve().parents[2]), location
assert package.__version__ == installed.version
if expected := os.environ.get("EXPECTED_VERSION"):
    assert installed.version == expected, (installed.version, expected)

smiles = "CCOC(=O)c1ccccc1"
assert dm.to_smiles(dm.to_mol(package.decode(package.encode(smiles)))) == smiles
print(f"safe-mol {installed.version}: {location}")
