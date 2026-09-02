<h1 align="center">  :safety_vest: SAFE </h1>
<h4 align="center"><b>S</b>equential <b>A</b>ttachment-based <b>F</b>ragment <b>E</b>mbedding (SAFE) is a novel molecular line notation that represents molecules as an unordered sequence of fragment blocks to improve molecule design using generative models.</h4>

</br>
<div align="center">
    <img src="docs/assets/safe-tasks.svg" width="100%">
</div>
</br>

<p align="center">
    <a href="https://arxiv.org/pdf/2310.10773.pdf" target="_blank">
      Paper
  </a> |
  <a href="https://safe-docs.datamol.io/" target="_blank">
      Docs
  </a> |
  <a href="https://huggingface.co/datamol-io/safe-gpt" target="_blank">
    🤗 Model
  </a> |
  <a href="https://huggingface.co/datasets/datamol-io/safe-gpt" target="_blank">
    🤗 Training Dataset
  </a>
</p>

---

</br>

[![PyPI](https://img.shields.io/pypi/v/safe-mol)](https://pypi.org/project/safe-mol/)
[![Conda](https://img.shields.io/conda/v/conda-forge/safe-mol?label=conda&color=success)](https://anaconda.org/conda-forge/safe-mol)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/safe-mol)](https://pypi.org/project/safe-mol/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/safe-mol)](https://anaconda.org/conda-forge/safe-mol)
[![Code license](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-red.svg)](DATA_LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/datamol-io/safe)](https://github.com/datamol-io/safe/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/datamol-io/safe)](https://github.com/datamol-io/safe/network/members)
[![arXiv](https://img.shields.io/badge/arXiv-2310.10773-b31b1b.svg)](https://arxiv.org/pdf/2310.10773.pdf)

[![test](https://github.com/datamol-io/safe/actions/workflows/test.yml/badge.svg)](https://github.com/datamol-io/safe/actions/workflows/test.yml)
[![release](https://github.com/datamol-io/safe/actions/workflows/release.yml/badge.svg)](https://github.com/datamol-io/safe/actions/workflows/release.yml)
[![code-check](https://github.com/datamol-io/safe/actions/workflows/code-check.yml/badge.svg)](https://github.com/datamol-io/safe/actions/workflows/code-check.yml)
[![doc](https://github.com/datamol-io/safe/actions/workflows/doc.yml/badge.svg)](https://github.com/datamol-io/safe/actions/workflows/doc.yml)

## Overview of SAFE

SAFE _is the_  deep learning molecular representation. It's an encoding leveraging a peculiarity in the decoding schemes of SMILES, to allow representation of molecules as a contiguous sequence of connected fragments. SAFE strings are valid SMILES strings, and thus are able to preserve the same amount of information. The intuitive representation of molecules as an ordered sequence of connected fragments greatly simplifies the following tasks often encountered in molecular design:

- _de novo_ design
- superstructure generation
- scaffold decoration
- motif extension
- linker generation
- scaffold morphing.

The construction of a SAFE strings requires defining a molecular fragmentation algorithm. By default, we use [BRICS], but any other fragmentation algorithm can be used. The image below illustrates the process of building a SAFE string. The resulting string is a valid SMILES that can be read by [datamol](https://github.com/datamol-io/datamol) or [RDKit](https://github.com/rdkit/rdkit).

</br>
<div align="center">
    <img src="docs/assets/safe-construction.svg" width="100%">
</div>

## Updates

SAFE 1.0 is a maintenance-focused major release. It preserves E/Z and atom
stereochemistry across fragmentation, makes strict and permissive decoding
behaviour explicit, supports extended ring closures, and updates SAFE-GPT to
Transformers 5 without changing the established seeded generation paths. The
core notation package is lightweight, while model, training, visualization and
Weights & Biases support are independent extras. Sampling gains an optional
`try_hard` quality pass and deterministic handling of linker and pattern
constraints.

See the [complete changelog](CHANGELOG.md) and the
[1.0 migration guide](docs/migration.md). These notes describe the upcoming
major release; PyPI and conda-forge still provide the published stable versions.

## News 🚀

#### 💥 2024/01/15 💥
1. [@IanAWatson](https://github.com/IanAWatson) has a C++ implementation of SAFE in [LillyMol](https://github.com/IanAWatson/LillyMol/tree/bazel_version_float) that is quite fast and use a custom fragmentation algorithm. Follow the installation instruction on the repo and checkout the docs of the CLI here: [docs/Molecule_Tools/SAFE.md](https://github.com/IanAWatson/LillyMol/blob/bazel_version_float/docs/Molecule_Tools/SAFE.md)


Release maintainers: see the [manual release guide](docs/releasing.md).

## Installation

SAFE 1.0 supports Python 3.11 through 3.14. Add it to a uv-managed project:

```bash
uv add safe-mol
```

Pip and conda-forge remain supported:

```bash
pip install safe-mol
mamba install -c conda-forge safe-mol
```

SAFE's core install contains only encoding, decoding and notation splitting.
Add `safe-mol[model]` for `SAFETokenizer` and `SAFEDesign`,
`safe-mol[train]` for the model stack plus `safe-train`, or
`safe-mol[all]` to install every maintained feature. Model APIs retain their
top-level imports but load their dependencies only when used. For example:

```bash
uv add "safe-mol[model]"
# or: python -m pip install "safe-mol[model]"
```

Visualization and Weights & Biases remain independently available through
`safe-mol[viz]` and `safe-mol[wandb]`. SAFE's optional model stack uses
Transformers 5. SAFE maintains random, greedy, beam and beam-sampling paths.
The constrained beam backend required by model-only linker generation is loaded
lazily from a reviewed, commit-pinned Hugging Face repository. RDKit 2026.03 is
excluded because of an upstream stereochemistry regression; RDKit 2024.09
through 2025.09 are covered by CI. See the
[1.0 migration guide](docs/migration.md) for details.

For GPU workloads, install the PyTorch build matching your CUDA driver before installing SAFE. You can verify the resulting environment with:

```python
import torch

print(torch.cuda.is_available())
```

### Datasets and Models

| Type                   | Name                                                                           | Infos      | Size  | Comment              |
| ---------------------- | ------------------------------------------------------------------------------ | ---------- | ----- | -------------------- |
| Model                  | [datamol-io/safe-gpt](https://huggingface.co/datamol-io/safe-gpt)              | 87M params | 350M  | Default model        |
| Training Dataset       | [datamol-io/safe-gpt](https://huggingface.co/datasets/datamol-io/safe-gpt)     | 1.1B rows  | 250GB | Training dataset     |
| Drug Benchmark Dataset | [datamol-io/safe-drugs](https://huggingface.co/datasets/datamol-io/safe-drugs) | 26 rows    | 20 kB | Benchmarking dataset |

## Usage

Please refer to the [documentation](https://safe-docs.datamol.io/), which contains tutorials for getting started with `safe` and detailed descriptions of the functions provided, as well as an example of how to get started with SAFE-GPT.

### API

We summarize some key functions provided by the `safe` package below.

| Function      | Description                                                                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `safe.encode` | Translates a SMILES string into its corresponding SAFE string.                                                                                                                                         |
| `safe.decode` | Translates a SAFE string into its corresponding SMILES string. The SAFE decoder just augment RDKit's `Chem.MolFromSmiles` with an optional correction argument to take care of missing hydrogen bonds. |
| `safe.split`  | Tokenizes a SAFE string to build a generative model.                                                                                                                                                   |

### Examples

#### Translation between SAFE and SMILES representations

```python
import safe

ibuprofen = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"

# SMILES -> SAFE -> SMILES translation
try:
    ibuprofen_sf = safe.encode(ibuprofen)  # c12ccc3cc1.C3(C)C(=O)O.CC(C)C2
    ibuprofen_smi = safe.decode(ibuprofen_sf, canonical=True)  # CC(C)Cc1ccc(C(C)C(=O)O)cc1
except safe.SAFEEncodeError:
    pass
except safe.SAFEDecodeError:
    pass

ibuprofen_tokens = list(safe.split(ibuprofen_sf))
```

### Training/Finetuning a (new) model

A command line interface is available to train a new model, please run `safe-train --help`. You can also provide an existing checkpoint to continue training or finetune on you own dataset.

For example:

```bash
safe-train --config <path to config> \
    --model-path <path to model> \
    --tokenizer  <path to tokenizer> \
    --dataset <path to dataset> \
    --num_labels 9 \
    --torch_compile True \
    --optim "adamw_torch" \
    --learning_rate 1e-5 \
    --prop_loss_coeff 1e-3 \
    --gradient_accumulation_steps 1 \
    --output_dir "<path to outputdir>" \
    --max_steps 5
```

## References

If you use this repository, please cite the following related [paper](https://arxiv.org/abs/2310.10773#):

```bib
@misc{noutahi2023gotta,
      title={Gotta be SAFE: A New Framework for Molecular Design},
      author={Emmanuel Noutahi and Cristian Gabellini and Michael Craig and Jonathan S. C Lim and Prudencio Tossou},
      year={2023},
      eprint={2310.10773},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

The Python code is licensed under [Apache-2.0](LICENSE). The training dataset
is separately licensed under [CC BY 4.0](DATA_LICENSE). These two files cover
different materials; they are not alternative licences for the Python package.

Note that the model weights of **SAFE-GPT** are exclusively licensed for research purposes (CC BY-NC 4.0).

The [SAFE-GPT model card](https://huggingface.co/datamol-io/safe-gpt/blob/3d5fa0988383e898d5ac5db7cd52bf715bc37061/README.md)
currently declares Apache-2.0 instead. This discrepancy needs maintainer
clarification before release; do not infer commercial-use permission from
the Python package licence. This repository does not redistribute the weights.

## Development lifecycle

### Setup dev environment

```bash
uv sync --all-extras
```

This creates an isolated `.venv` with the training, visualisation, reporting,
test, documentation and development extras. `env.yml` remains available when a
Conda environment is required.

### Tests

You can run tests locally with:

```bash
uv run python -m pytest -m "not integration"
uv run python -m pytest -m integration --no-cov
```

The integration command validates the published SAFE-GPT model and executes
the maintained tutorials. GitHub Actions runs the same command. Use
`uv run python -m pytest -m notebook --no-cov` when iterating on tutorials only.
