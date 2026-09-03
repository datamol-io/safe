# SAFE Changelog

This file records user-visible changes. See the [migration guide](docs/migration.md)
for upgrade instructions and [GitHub releases](https://github.com/datamol-io/safe/releases)
for earlier release notes.

## Next major release (unreleased)

### Highlights

- Preserve molecular stereochemistry and graph identity more rigorously across
  SAFE fragmentation and reconstruction.
- Move SAFE-GPT generation and training to Transformers 5 while preserving the
  established logits and seeded sampling behavior.
- Make the core notation package lightweight and isolate model, training,
  visualization and experiment-reporting dependencies in optional extras.

### Added

- Add support for RDKit extended ring closures such as `%(100)` in encoding,
  decoding and tokenization.
- Add `allow_empty=True` to `safe.encode`. Inputs the slicer cannot cut (rigid
  rings, single atoms, or the components of a salt) are returned as a single
  unfragmented SAFE block instead of raising `SAFEFragmentationError`.
- Add `try_hard=True` to every public design workflow. It oversamples,
  validates, checks the requested structural constraint, removes duplicates in
  generation order and returns at most the requested number of candidates.
- Add `model_revision` and `model_dir` controls for reviewed Hub revisions and
  offline or local SAFE-GPT checkpoints.
- Add deterministic regression tests for greedy, multinomial, beam,
  beam-sampling and constrained linker generation.

### Changed

- Clarify the existing licensing boundaries: Apache-2.0 for code, CC BY 4.0
  for the training dataset, and research-only CC BY-NC 4.0 for SAFE-GPT weights,
  as confirmed by the maintainer. The package does not redistribute the weights.
- Test Python 3.11–3.14 and RDKit 2024.09, 2025.03 and 2025.09. The core
  requires Python 3.11+ and RDKit 2024.09+; model extras require PyTorch 2.5+.
  Mac Intel supports the notation core, not the model/training extras, because
  official PyTorch wheels for that platform stop at 2.2.
  RDKit 2026.03 is excluded because its current fragmentation direction
  handling can silently alter double-bond stereochemistry.
- Support Transformers 5 for model and training features. The constrained-beam
  implementation required by model-only linker generation is loaded lazily
  from a reviewed, commit-pinned Hugging Face repository; ordinary sampling
  never downloads it.
- Make `safe-mol` the encoding, decoding and splitting core. Use the `model`,
  `train`, `viz`, `wandb` or `all` extras only for the corresponding features.
- Make canonical decoding a serialization choice only. It no longer changes
  charges or selects a canonical tautomer.
- Use `max_new_tokens` as the prompt-independent generation budget.
- Pin the default SAFE-GPT revision and use stable candidate ordering and
  operation-local random generators.

### Deprecated

- Deprecate `motif_extension()`, which is an exact alias of
  `scaffold_decoration(..., add_dot=True)`. It is scheduled for removal in
  SAFE 2.0.
- Deprecate `max_length` in public sampling helpers in favor of
  `max_new_tokens`; passing both is an error.

### Removed

- Remove SAFE wrappers for contrastive and diverse beam search, which are no
  longer part of the maintained Transformers 5 generation surface. Advanced
  experiments can call the underlying model directly.
- Remove PyTorch, Transformers, Tokenizers, tqdm and fsspec from the default
  notation-only installation.

### Fixed

- Preserve E/Z configuration, directional bonds shared by multiple double
  bonds, tetrahedral and supported non-tetrahedral atom stereochemistry, and
  explicit hydrogens required for stereo. When no safe cut exists, encoding
  returns the exact unfragmented molecule.
- Reject enhanced CXSMILES stereo groups explicitly rather than silently losing
  their AND, OR or absolute-group semantics.
- Make canonical encoding invariant to equivalent input SMILES and verify the
  final dummy-aware isomeric graph.
- Keep terminal wildcard attachment points open without changing lone or
  degree-two wildcard topology.
- Make strict decoding raise `SAFEDecodeError`; retain `ignore_errors=True` as
  the batch-friendly path returning `None` for invalid entries.
- Preserve the batch dimension for one-item property predictions and support
  the current Transformers trainer signature.
- Save a loadable tokenizer, restore the SAFE-aware pre-tokenizer after
  pickling, and retain extended ring-closure tokens.
- Correct model-only linker constraints, remove leaked intermediate SAFE
  strings, and make linker, scaffold and pattern sampling deterministic across
  Python hash seeds and CPU, CUDA or Apple Silicon MPS devices.
- Fix model-only linker/scaffold-morphing fragment selection: the cut kept only
  the first generated fragment, dropping the fragment that carried the required
  ring closure when the model emitted it later. It now retains fragments up to
  and including the closure-bearing one.
- Sanitize pattern exemplars before completion, report unavailable linkers
  consistently, validate visualization modes and require an explicit tokenizer
  in the training CLI.
- Make the design workflows behave consistently: `super_structure` now verifies
  the requested core is present when sanitizing (like the other constrained
  methods), `filter_by_substructure_constraints` normalizes `dm.Mol` queries the
  same way as strings so attachment points act as wildcards, every workflow
  guards `n_trials` and finalizes through the same code path, and `try_hard`
  warns instead of silently returning fewer candidates than requested.

### Compatibility and delivery

- Validate the minimal core, every maintained extra, SAFE-GPT, training CLI and
  executable tutorials in separate CI jobs.
- Cover the core on Linux, Windows, macOS Apple Silicon and macOS Intel across
  the supported Python and RDKit matrix; run model integration tests on Linux.
- Keep publication manual through the `release` action and `PYPI_API_TOKEN`,
  with PEP 740 attestations. Release tests and isolated wheel/source checks
  gate publication; prereleases never replace the stable documentation.
- Add a non-publishing dry run and a [release guide](docs/releasing.md).
  Conda-forge remains a separate channel requiring recipe updates.
