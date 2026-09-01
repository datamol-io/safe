# Migrating to SAFE 1.0

SAFE 1.0 is a maintenance-focused major release. It keeps the established encoding, decoding, tokenization, generation, and training behaviour while moving the supported environment and delivery process forward.

## Supported environment

- Python 3.11 through 3.14 is supported. Python 3.9 and 3.10 are no longer tested.
- The minimum RDKit release is 2024.09. RDKit 2026.03 is deliberately excluded because that series changes double-bond direction handling during fragmentation and can silently lose stereochemistry in an otherwise valid SAFE round trip. The compatibility matrix uses RDKit 2024.09, 2025.03, and 2025.09.
- PyTorch 2.5 or newer is supported.
- Transformers 5 is the supported generation stack. Regression oracles cover greedy, multinomial, beam, beam-sampling, diverse-beam, constrained-beam, contrastive and prompt-lookup-assisted decoding. SAFE-GPT keeps identical logits and seeded outputs for the six established decoding paths compared with Transformers 4.57.6; contrastive search is restored through its pinned backend and matches that backend under both releases.
- Datasets 4+, Accelerate 1.1+, Tokenizers 0.23, and current NumPy and NetworkX releases are supported.

The current Transformers generation stack moved contrastive, constrained and
diverse beam search into custom generation
repositories. SAFE loads those algorithms only when requested, pins the exact
reviewed upstream commits, and allows offline mirrors through
`SAFE_CONTRASTIVE_GENERATION_BACKEND`,
`SAFE_CONSTRAINED_GENERATION_BACKEND` and
`SAFE_GROUP_BEAM_GENERATION_BACKEND`. Standard sampling does not download or
execute any of these backends.

The default installation is now the molecular notation core: encoding,
decoding and `safe.split`. PyTorch, Transformers, Tokenizers, tqdm and fsspec
move to the `model` extra. `SAFEDesign` and `SAFETokenizer` keep their public
top-level names and load that stack only when used. The `train` extra includes
the model stack plus Datasets, Evaluate, Accelerate and universal-pathlib.
Matplotlib and Weights & Biases remain isolated in `viz` and `wandb`; `all`
installs every maintained feature.

Recreate the environment rather than upgrading it in place:

```bash
uv sync --all-extras
```

`env.yml` remains a supported Conda alternative. For a pip training
environment, install `safe-mol[train]` (or `safe-mol[train,wandb]` if
experiment reporting is required).

For GPU installations, install the PyTorch build appropriate for the CUDA driver before installing SAFE.

## Behaviour-preserving fixes

- Canonical SAFE encoding is now invariant to equivalent input SMILES spellings.
  Passing `randomize=True` with `canonical=True` is explicitly a no-op, as the
  public API has always documented.
- Encoding with `ignore_stereo=False` no longer cuts stereogenic double bonds or
  the directional single bonds that define E/Z geometry. SAFE now verifies the
  complete isomeric graph before returning an encoding. If a custom slicer still
  changes specified stereochemistry, encoding raises `SAFEEncodeError` instead
  of returning a silently different stereoisomer. Set `ignore_stereo=True` only
  when dropping stereochemistry is intentional.
- `decode(..., canonical=True)` now means canonical SMILES serialization only.
  It no longer standardizes charges or selects a canonical tautomer, operations
  which could change the molecular graph represented by an otherwise valid SAFE
  string.
- Decode failures now follow the documented contract: strict decoding raises
  `SAFEDecodeError`, while `ignore_errors=True` remains the permissive batch path
  and returns `None` for an invalid item.
- Wildcard-containing molecules no longer turn a structural wildcard into an invalid ring closure. Explicitly labelled and terminal wildcard attachment points remain open for linker generation and scaffold morphing. Their SAFE form is an unmatched ring-closure token rather than a literal `*`; decode with `remove_dummies=False` when the attachment points must be visible again as wildcard atoms.
- Molecules with 100 or more attachment bonds use RDKit's extended `%(nnn)` ring-closure notation. The SAFE tokenizer now treats that notation as one token.
- A one-item batch keeps its batch dimension in the property head.
- `SAFETrainer.compute_loss` accepts the current Transformers trainer call signature.
- `SAFETokenizer.save_pretrained()` now writes a loadable `tokenizer.json` instead of calling a method that does not exist on the underlying Rust tokenizer.
- Pickling a `SAFETokenizer` now restores its SAFE-aware pre-tokenizer, including
  extended ring-closure tokens such as `%(100)`.
- Model-only linker constraints now encode each ring-closure permutation as a complete phrase. Previously, a dot token alone could satisfy the disjunctive constraint, and intermediate SAFE strings leaked into the returned SMILES list.
- Model-only linker candidates use a stable sorted order instead of iterating an
  unordered set, so results no longer depend on `PYTHONHASHSEED`.
- Pattern sampling allocates tensors on the model device (CPU, CUDA or Apple
  Silicon MPS) and uses operation-local random generators. Each generation trial
  also receives a distinct, reproducible SAFE randomization seed.
- Pattern decoration now converts sampled concrete scaffolds to sanitized
  molecules before completion and skips invalid exemplars, instead of passing a
  SMARTS query molecule into the SAFE encoder.
- Linker extraction reports an unavailable linker as `(molecule, None, None)`
  when the requested minimum size cannot be met. Visualization accepts an
  explicit fragment sequence and rejects unknown highlight modes with a clear
  error.
- The training CLI now reports a clear error when `--tokenizer` is missing.

## Sampling transition

`max_new_tokens` is now the default generation-length control because it gives
the completion the same budget regardless of prompt length:

```python
designer.scaffold_decoration(scaffold, max_new_tokens=80)
```

`max_length` retains its previous total-length semantics for this release and
emits a `FutureWarning`; it is scheduled for removal in SAFE 2.0. Passing both
length controls is an error.

All public design methods accept `try_hard=False`. Enabling it performs a
transparent quality pass: SAFE samples three times the requested candidates,
decodes and validates them, preserves any requested fragment or scaffold
constraint, removes duplicates in generation order, and returns at most the
requested count. It does not change model logits or apply medicinal-chemistry
heuristics:

```python
designer.linker_generation(
    "[*]c1ccccc1",
    "[*]N1CCCCC1",
    n_samples_per_trial=20,
    try_hard=True,
)
```

The published `datamol-io/safe-gpt` checkpoint is loaded at a reviewed pinned
revision by default. Pass `model_revision=` explicitly to test another Hub
revision, or `model_dir=` to load a local model.

## CI and releases

Pull requests and pushes to `dev` and `main` now run the supported Python/RDKit
matrix with `safe-mol[all]` on Linux x86-64, Windows x86-64, macOS Apple
Silicon and macOS Intel.
The published SAFE-GPT checkpoint and executable tutorials have a separate
Linux integration lane because they download external model artifacts.
Documentation is built in strict mode.

Releases are built once from a published GitHub Release, smoke-tested from both
the wheel and source distribution, supplied with PEP 740 attestations, and
uploaded to PyPI with Trusted Publishing. The release workflow no longer
creates tags or pushes directly to `main`. The conda-forge feedstock remains a
supported downstream channel; its update bot proposes each PyPI version and
maintainers review the resulting dependency and test changes there.
