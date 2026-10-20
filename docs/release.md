# Release guidance

This page describes preparation for a release. It is not a record of an upload, published package, assigned DOI, or author approval. Use the repository's executed validation report for actual checks and results; do not infer that the checklist below has been completed.

## Before making a release

1. Confirm the intended version, public API, supported Python versions, dependency declarations, and documentation match the built artifact.
2. Run the tests and documented synthetic examples in a clean environment. Record commands, versions, results, and any limitations in the validation report.
3. Build the source distribution and wheel, inspect their contents, and install the wheel into a separate clean environment. Check imports and a small synthetic diagnostic there.
4. Check that excluded local data, secrets, temporary files, private audit notes, and machine-specific paths are absent from distributable files.
5. The selected software license is BSD-3-Clause. Confirm the repository's `LICENSE` and built packaging metadata agree and retain the required notices in redistributions. The software license does not automatically license third-party data or separately published manuscripts.
6. Confirm attribution, equal contribution, AI-assistance disclosures, and the accuracy of any claimed publication or persistent identifier. Do not create a DOI, package-index URL, or repository URL by assumption.
7. Review numerical conventions and scope statements. In particular, retain Fisher's default-off gate, invalid-draw reporting, the transformed-variance target, and the empirical-application limits.
8. Record artifact hashes and release notes. Obtain authorization for the intended external publication step before uploading or pushing release artifacts.

## Build and verify locally

From a fresh checkout or extraction, create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
python -m pytest
python validation/reference_check.py
python validation/reproduce_paper_benchmarks.py --output validation/local_benchmarks.json
python examples/synthetic_panel.py
python examples/clustered_panel.py
python -m build
python -m twine check dist/*
```

The benchmark command replays only three original analytic scenarios (15,000 panels). It is optional for routine installation and is not the entire paper study. Exact bit equality is separately available with `--strict-bitwise` in the recorded environment; portable checking reports tolerance agreement independently.

Then create a second environment and install the built wheel, without importing from the source tree:

```sh
python3 -m venv ../didcheck-wheel-test
../didcheck-wheel-test/bin/python -m pip install dist/didcheck-0.1.0-py3-none-any.whl
../didcheck-wheel-test/bin/python -I -c "import didcheck; print(didcheck.__version__, didcheck.__file__)"
```

Use `Scripts/python.exe` instead of `bin/python` on Windows. The Python 3.10–3.13 matrix passed on Ubuntu for the initial release; see `VALIDATION_REPORT.md` for the executed run. Each later release must pass its own workflow.

## GitHub release and Trusted Publishing

Complete final author approval before making this repository public. Confirm the BSD-3-Clause permissions and preprint citation, then create the intended repository, add its verified URL to `pyproject.toml` and `CITATION.cff`, push the reviewed source, and require all Python matrix checks to pass. Update the README's installation status when publication is verified.

Build the final wheel and source distribution from the approved source and record both SHA-256 hashes in a two-line `SHA256SUMS.txt` file using distribution basenames. Create GitHub release `v0.1.0` and attach these two distributions and that checksum file. The full delivery ZIP has a different, larger manifest; do not use its manifest as the release-asset checksum file.

Configure a GitHub environment named `pypi`, with a required reviewer and suitable branch restrictions. In the PyPI account, register a pending GitHub Trusted Publisher for project `didcheck`, the actual repository owner, repository `DiDCheck`, workflow `publish.yml`, and environment `pypi`. Account ownership and approval must be established through the official services; never commit credentials.

After approval and registration, manually dispatch **Publish approved release to PyPI** with tag `v0.1.0`. It downloads and checks the existing GitHub release assets, then uploads those same bytes with OIDC. It does not rebuild the distributions. Publishing is not triggered by pushes or by creating a release. See the [official PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

Verify the public PyPI JSON listing and file hashes, then install `didcheck==0.1.0` with `--no-cache-dir --index-url https://pypi.org/simple` in a fresh virtual environment. Run an import, a synthetic diagnostic, and the documented examples. Record the public URLs and actual CI results. A missing PyPI project response does not reserve a name or guarantee acceptance; recheck before upload.

## Future updates

Change the package version in `pyproject.toml`, `src/didcheck/__init__.py`, citation metadata, and release notes together. Update the exact tag and distribution-name guards in `publish.yml` for the intended release. Repeat local validation and GitHub CI, obtain release approval, create a new tag and release with checked distributions, then manually publish. Never reuse a published version for different bytes. Review Trusted Publisher and environment access when maintainers change.

## Version changes that affect scientific use

Call out changes to demeaning, period ordering, sample selection, variance divisors, covariance construction, rank tolerance, cluster multipliers, resampling units, centering, rejection inequalities, or random-number generation. Even apparently small changes can alter results or exact seeded replay.

Preserve enough provenance to associate a reported result with its data-preparation choices, software version, sampling design, random seed, and bootstrap draw count. A compatible function signature alone does not establish numerical equivalence to an older implementation.

## Claims supported by a release

Report software checks as software checks. A deterministic calculation can verify arithmetic; a replay can verify reproducibility; a Monte Carlo study can assess selected designs with Monte Carlo uncertainty. None proves all sampling assumptions, exact finite-sample size, or validity of an empirical application.

Do not advertise the package as a parallel-trends certificate, an estimator of treatment effects, or a validated China Shock or minimum-wage analysis. Publishing this repository would not change those limits.
