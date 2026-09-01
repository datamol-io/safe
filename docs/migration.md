# Migrating to SAFE 1.0

SAFE 1.0 is a maintenance-focused major release. It keeps the established encoding, decoding, tokenization, generation, and training behaviour while moving the supported environment and delivery process forward.

## Supported environment

- Python 3.11 through 3.14 is supported. Python 3.9 and 3.10 are no longer tested.
- The minimum RDKit release is 2024.09. RDKit 2026.03 is deliberately excluded because that series changes double-bond direction handling during fragmentation and can silently lose stereochemistry in an otherwise valid SAFE round trip. The compatibility matrix uses RDKit 2024.09, 2025.03, and 2025.09.
- PyTorch 2.5 or newer is supported.
- Transformers 4.57 is the supported generation stack. Transformers 5 is intentionally excluded because it removed `PhrasalConstraint` and `DisjunctiveConstraint`, which SAFE uses for constrained linker generation and scaffold decoration. Upgrading prematurely would remove existing behaviour rather than modernize it.
- Datasets 4+, Accelerate 1.1+, Tokenizers 0.22, and current NumPy and NetworkX releases are supported.

Recreate the environment rather than upgrading it in place:

```bash
micromamba create -f env.yml
micromamba activate safe
```

For GPU installations, install the PyTorch build appropriate for the CUDA driver before installing SAFE.

## Behaviour-preserving fixes

- Wildcard-containing molecules no longer turn a structural wildcard into an invalid ring closure. Explicitly labelled and terminal wildcard attachment points remain open for linker generation and scaffold morphing. Their SAFE form is an unmatched ring-closure token rather than a literal `*`; decode with `remove_dummies=False` when the attachment points must be visible again as wildcard atoms.
- Molecules with 100 or more attachment bonds use RDKit's extended `%(nnn)` ring-closure notation. The SAFE tokenizer now treats that notation as one token.
- A one-item batch keeps its batch dimension in the property head.
- `SAFETrainer.compute_loss` accepts the current Transformers trainer call signature.
- `SAFETokenizer.save_pretrained()` now writes a loadable `tokenizer.json` instead of calling a method that does not exist on the underlying Rust tokenizer.
- The training CLI now reports a clear error when `--tokenizer` is missing.

## CI and releases

Pull requests and pushes to `dev` and `main` now run the supported Python/RDKit matrix on Linux, with additional macOS and Windows lanes. The published SAFE-GPT checkpoint and executable tutorials have a separate integration lane. Documentation is built in strict mode.

Releases are built once from a published GitHub Release, smoke-tested from the wheel, and uploaded to PyPI with Trusted Publishing. The release workflow no longer creates tags or pushes directly to `main`.
