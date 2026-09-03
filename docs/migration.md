# Migrating to SAFE 0.2.0

SAFE 0.2.0 is a maintenance-focused release. It keeps the established encoding, decoding, tokenization, generation, and training behaviour while moving the supported environment and delivery process forward.

## Supported environment

- Python 3.11 through 3.14 is supported. Python 3.9 and 3.10 are no longer tested.
- The minimum RDKit release is 2024.09. RDKit 2026.03 is deliberately excluded because that series changes double-bond direction handling during fragmentation and can silently lose stereochemistry in an otherwise valid SAFE round trip. The compatibility matrix uses RDKit 2024.09, 2025.03, and 2025.09.
- Model and training extras require PyTorch 2.5 or newer. Official macOS Intel
  wheels stop at PyTorch 2.2, so those extras are not supported natively there.
  The notation core works without PyTorch, including alongside Molfeat on Intel.
- Transformers 5 is the supported generation stack. SAFE maintains greedy,
  multinomial, beam, beam-sampling and the constrained-beam path required by
  model-only linker generation. SAFE-GPT keeps identical logits and seeded
  outputs for those established paths compared with Transformers 4.57.6.
- Datasets 4+, Accelerate 1.1+, Tokenizers 0.23, and current NumPy and NetworkX releases are supported.

The current Transformers generation stack moved constrained beam search into a
custom generation repository. SAFE loads it only for model-only linker
generation, pins the reviewed upstream commit, and allows an offline mirror
through `SAFE_CONSTRAINED_GENERATION_BACKEND`. Standard sampling does not
download or execute this backend. Contrastive and diverse beam search are no
longer wrapped by SAFE; advanced Transformers experiments should call the
underlying model directly.

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
- Encoding with `ignore_stereo=False` no longer cuts stereogenic double bonds,
  directional single bonds shared by multiple E/Z definitions, or bonds incident
  to specified non-tetrahedral atom stereocentres. It also protects unsafe
  explicit-hydrogen cuts near atom stereocentres. Directional single bonds local
  to one double bond remain cuttable when the exact isomeric graph round-trips.
  When every candidate cut is unsafe, or the final fragmented graph changes the
  stereo assignment, SAFE returns the exact molecule unfragmented rather than
  rejecting a valid `safe.encode()` call. A final dummy-aware isomeric graph
  comparison catches changes that standard InChI identity can miss. Enhanced
  CXSMILES stereo groups are rejected explicitly in
  SAFE 0.2.0 because plain SAFE/SMILES cannot retain their AND, OR or absolute-group
  semantics; resolve the group to one stereoisomer, or set `ignore_stereo=True`
  only when dropping that information is intentional.
- `decode(..., canonical=True)` now means canonical SMILES serialization only.
  It no longer standardizes charges or selects a canonical tautomer, operations
  which could change the molecular graph represented by an otherwise valid SAFE
  string.
- Decode failures now follow the documented contract: strict decoding raises
  `SAFEDecodeError`, while `ignore_errors=True` remains the permissive batch path
  and returns `None` for an invalid item.
- Wildcard-containing molecules no longer turn a structural wildcard into an invalid ring closure. Terminal wildcard attachment points remain open for linker generation and scaffold morphing, while degree-two and lone wildcards keep their original topology. An open attachment's SAFE form is an unmatched ring-closure token rather than a literal `*`; decode with `remove_dummies=False` when attachment points must be visible again as wildcard atoms.
- Molecules with 100 or more attachment bonds use RDKit's extended `%(nnn)` ring-closure notation. The SAFE tokenizer accepts every RDKit extended form from `%(1)` through five digits as one token.
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

`motif_extension()` was an exact wrapper around
`scaffold_decoration(..., add_dot=True)`. It remains as a deprecated alias for
this transition and will be removed in SAFE 2.0.

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
core matrix on Linux x86-64, Windows x86-64, macOS Apple Silicon and macOS
Intel. All maintained extras, the published SAFE-GPT checkpoint and executable
tutorials are validated in a separate Linux integration lane.
Documentation is built in strict mode.

Publication remains a manual action, using PyPI Trusted Publishing (OpenID
Connect) with full test validation and isolated wheel/source installation checks. The action creates
a GitHub tag and Release at the tested commit only after PyPI succeeds; it
never pushes code to `main`. See the [release guide](releasing.md) for dry
runs, prereleases and the separate conda-forge recipe updates.
