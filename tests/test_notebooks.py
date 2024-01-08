import pathlib

import nbformat
import pytest
from nbconvert.preprocessors.execute import ExecutePreprocessor

ROOT_DIR = pathlib.Path(__file__).parent.resolve()

TUTORIALS_DIR = ROOT_DIR.parent / "docs" / "tutorials"
DISABLE_NOTEBOOKS = []
NOTEBOOK_PATHS = sorted(TUTORIALS_DIR.glob("*.ipynb"))
NOTEBOOK_PATHS = list(filter(lambda x: x.name not in DISABLE_NOTEBOOKS, NOTEBOOK_PATHS))

# Discard some notebooks
NOTEBOOKS_TO_DISCARD = ["extracting-representation-molfeat.ipynb"]
NOTEBOOK_PATHS = list(filter(lambda x: x.name not in NOTEBOOKS_TO_DISCARD, NOTEBOOK_PATHS))


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=[str(n.name) for n in NOTEBOOK_PATHS])
def test_notebook(nb_path):
    # Setup and configure the processor to execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    # Open the notebook
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # Execute the notebook
    ep.preprocess(nb, {"metadata": {"path": TUTORIALS_DIR}})
