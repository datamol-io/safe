
<h1 align="center">  :safety_vest: SAFE </h1>
<h4 align="center">S</b>equential <b>A</b>ttachment-based <b>F</b>ragment <b>E</b>mbedding (SAFE) is a novel molecular line notation that represents molecules as an unordered sequence of fragment blocks to improve molecule design using generative models.</h4>

</br>
<div align="center">
    <img src="assets/safe-tasks.svg" width="100%">
</div>
<p align="center">
    <a href="" target="_blank">
      Paper
  </a> |
  <a href="https://github.com/valence-labs/safe/" target="_blank">
      Github
  </a> |
  <a href="#" target="_blank">
    🤗 Model
  </a>
</p>

---


## 🆕 News
- \[**September 2023**\] We've released a SAFE GPT-like pretrained model on a combination of ZINC and UniChem


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
    <img src="assets/safe-construction.svg" width="100%">
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

We provided a pretained GPT2 model (XXM parameters) using the SAFE molecular representation that has been trained on 1.1 billion molecules from Unichem (0.1B) + Zinc (1B): 

- *Safe-XXM* [maclandrol/safe-XXM]()


### Usage

To get started with SAFE, please see the tutorials: 
- xxx
- xxx


## References
If you use this repository, please cite the following related paper:

```
@article{,
  title={Gotta be SAFE: a new framework for molecular design.},
  author={},
  journal={},
  year={2023}
}
```

## License

Please note that all data and model weights of **SAFE** are exclusively licensed for research purposes. The accompanying dataset is licensed under CC BY 4.0, which permits solely non-commercial usage. See [DATA_LICENSE](DATA_LICENSE) for details.

This code base is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.
