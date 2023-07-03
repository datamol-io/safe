
<h1 align="center">  :safety_vest: `SAFE` </h1>
<h4 align="center">S</b>equential <b>A</b>ttachment-based <b>F</b>ragment <b>E</b>mbedding (SAFE) is a novel molecular line notation that represents molecules as an unordered sequence of fragment blocks to improve molecule design using generative models.<h4>

</br>
<div align="center">
    <img src="assets/safe-tasks.svg" width="100%">
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


## 🆕 News
- \[**August 2023**\] We've released xxx


## Overview of SAFE

SAFE is a 

### Installation

You can install `safe` using pip.

```bash
pip install safe-smiles
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
````

## License

Please note that all data and model weights of **SAFE** are exclusively licensed for research purposes. The accompanying dataset is licensed under CC BY 4.0, which permits solely non-commercial usage. See [DATA_LICENSE](DATA_LICENSE) for details.

This code base is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.
