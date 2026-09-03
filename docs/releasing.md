# Releasing SAFE

Publication is manual. Merging into `main`, pushing a tag or publishing a
GitHub Release does not upload a package to PyPI.

## Before publishing

1. Finalize the release notes in `CHANGELOG.md` with an exact version heading,
   for example `## 1.0.0 - YYYY-MM-DD`. Date the release and update README
   and website wording that still calls it unreleased or specific to `dev`.
2. Merge the release changes into `main`, preserving contributor history.
3. Confirm the PyPI Trusted Publisher for `safe-mol` is registered for this
   repository, the `release.yml` workflow and the `pypi` environment. No API
   token is used; the upload authenticates over OpenID Connect.

## Run the release action

Open **Actions → release → Run workflow**. Select `main`, enter
`1.0.0` in **release-version**, and leave **dry-run** checked for a
rehearsal. Dry runs may also run from `dev` and accept unfinished release
notes. They create a tag only inside the temporary runner checkout, never on
GitHub, and cannot publish packages or documentation.

For publication, launch the action again from `main` with **dry-run**
unchecked. Use canonical versions without a `v` prefix, such as `1.0.0`
or `1.0.0rc1`; `a`, `b` and `rc` suffixes identify prereleases.

The action validates the version and release notes, reruns the complete test
and quality workflows on the selected commit, and builds the distributions.
Both wheel and source installations are checked with Python's isolated mode,
including their version and import location. Documentation must also build
successfully before anything is uploaded.

The publish job uploads the artifacts and PEP 740 attestations to PyPI over
GitHub's OpenID Connect (Trusted Publishing); no API token is used. This
requires a PyPI Trusted Publisher registered for this repository, the
`release.yml` workflow and the `pypi` environment.

Only after PyPI succeeds does the action create the GitHub tag and Release
at the tested commit, then deploy versioned documentation. A prerelease never
moves the `stable` documentation alias. A failed upload leaves that alias
unchanged. If a later step fails, rerun the failed jobs rather than rebuilding
an already published version.

## Conda-forge

The [feedstock](https://github.com/conda-forge/safe-mol-feedstock) is a separate
release channel. Its update bot proposes version changes after PyPI publication,
but maintainers must review dependencies and run the recipe tests there.
The package release action does not publish to conda-forge.

Raise Python to 3.11+, exclude RDKit 2026.03, and remove model and training
dependencies from the core recipe. Move the `safe-train --help` recipe test to
a training-enabled environment; it cannot run with only the notation core.

Do not merge a version-only feedstock update for this release. Its
dependency metadata and import/CLI tests must reflect the new installation
boundaries.
