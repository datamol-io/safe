
<h1 align="center">  :safety_vest: `SAFE` </h1>
<h4 align="center">S</b>equential <b>A</b>ttachment-based <b>F</b>ragment <b>E</b>mbedding (SAFE) is a novel molecular line notation that represents molecules as an unordered sequence of fragment blocks to improve molecule design using generative models.<h4>

</br>
<div align="center">
    <img src="docs/assets/safe-tasks.svg" width="100%">
</div>
<p align="center">
    <a href="" target="_blank">
      Paper
  </a> |
  <a href="https://maclandrol.github.io/safe/" target="_blank">
      Docs
  </a> |
  <a href="#" target="_blank">
    🤗 Model
  </a>
</p>

---

[![PyPI](https://img.shields.io/pypi/v/safe)](https://pypi.org/project/safe/)
[![Version](https://img.shields.io/pypi/pyversions/safe)](https://pypi.org/project/safe/)
[![Code license](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/maclandrol/safe/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-red.svg)](https://github.com/maclandrol/safe/blob/main/DATA_LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://github.com/maclandrol/safe/graphs/commit-activity)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
[![test](https://github.com/maclandrol/safe/actions/workflows/test.yml/badge.svg)](https://github.com/maclandrol/safe/actions/workflows/test.yml)

## 🆕 News
- \[**August 2023**\] We've released xxx


## Overview of SAFE

SAFE *is the* deep learning molecular representation. It's an encoding leveraging a peculiarity in the decoding schemes of SMILES, to allow representation of molecules as contiguous sequence of connected fragment. SAFE strings are valid SMILES string, and thus are able to preserve the same amount of information.  The intuitive representation of molecules as unordered sequence of connected fragments gretly simplify the following tasks often encoutered in molecular design:

- *de novo* design
- superstructure generation
- scaffold decoration
- motif extension
- linker generation
- scaffold morphing. 

The construction of a SAFE strings requires definition a molecular fragmentation algorithm. By default, we use [BRICS], but any other fragmentation algorithm can be used. The image below illustrate the process of building a SAFE string. The resulting string is a valid SMILES that can be read by [datamol](https://github.com/datamol-io/datamol) or [RDKit](https://github.com/rdkit/rdkit).

</br>
<div align="center">
    <img src="docs/assets/safe-construction.svg" width="100%">
</div>



### Installation

You can install `safe` using pip.

```bash
pip install safe-mol
```

Alternatively clone this repo, install the dependencies, install `safe` locally and you are good to go:


```bash
git clone https://github.com/maclandrol/safe.git
cd safe
mamba env create -f env.yml -n "safe-space" # :)
pip install -e .
```

`safe` mostly depends on [transformers](https://huggingface.co/docs/transformers/index) and [datasets](https://huggingface.co/docs/datasets/index). Please see the [env.yml](./env.yml) file for a complete list of dependencies.


### Datasets and Models

We provided a pretained GPT2 model (50M parameters) using the SAFE molecular representation that has been trained on 1.1 billion molecules from Unichem (0.1B) + Zinc (1B): 

- *Safe-1.1B-dataset* [maclandrol/safe-50M]()
- *Safe-50M* [maclandrol/safe-50M]()


## Usage

Please refer to the [documentation](), which contains a thorough tutorial for getting started with ``safe`` and detailed descriptions of the functions provided. 

In particular, see the following tutorials:
- xxx
- xxx


### API

We summarize some key functions provided by the `safe` package below.

| Function        | Description                                                                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ``safe.encode`` | Translates a SMILES string into its corresponding SAFE string.                                                                                                                                          |
| ``safe.decode`` | Translates a SAFE string into its corresponding SMILES string. The SAFE decoder just augment RDKit's `Chem.MolFromSmiles` with an optional correction argument to take care of missing hydrogens bonds. |
| ``safe.split``  | Tokenizes a SAFE string to build a generative model.                                                                                                                                                    |


### Examples

#### Translation between SAFE and SMILES representations

```python
import safe

ibuprofen = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"

# SMILES -> SAFE -> SMILES translation
try:
    ibuprofen_sf = safe.encode(ibuprofen)  # [C][=C][C][=C][C][=C][Ring1][=Branch1]
    ibuprofen_smi = safe.decode(ibuprofen_sf, canonical=True)  # CC(Cc1ccc(cc1)C(C(=O)O)C)C
except safe.EncoderError:
    pass 
except safe.DecoderError:
    pass

ibuprofen_tokens = list(safe.split(ibuprofen_sf))
# ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[=Branch1]']
```


## Changelog
See the latest changelogs at [CHANGELOG.rst](./CHANGELOG.rst).

## References
If you use this repository, please cite the following related paper:

```
@article{,
  title={Gotta be SAFE: a new framework for molecular design.},
  author={},
  journal={},
  year={2023}
}
````

## License

Please note that all data and model weights of **SAFE** are exclusively licensed for research purposes. The accompanying dataset is licensed under CC BY 4.0, which permits solely non-commercial usage. See [DATA_LICENSE](DATA_LICENSE) for details.

This code base is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.

## Maintainers

- @maclandrol
