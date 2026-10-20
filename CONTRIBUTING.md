# Contributing to DiDCheck

Describe the problem, proposed behavior, and scientific impact in an issue or pull request. Use synthetic examples; do not upload personal data, confidential datasets, credentials, or materials you cannot redistribute.

Create a virtual environment and run `python -m pip install ".[dev]"`, `python -m pytest`, `python validation/reference_check.py`, and `python validation/reproduce_paper_benchmarks.py`. Run both scripts in `examples/` for changes to the public API or examples. See `docs/release.md` for packaging checks.

Preserve the frozen statistical core and its provenance. Any intentional scientific change needs an explicit rationale, appropriate tests, updated documentation, and a changelog entry. Do not update reference fixtures merely to make tests pass. Report failures and changes to random-number generation or inference conventions explicitly.

Contributions are proposed under the repository's BSD-3-Clause license. Submit only material you are entitled to license. The license does not cover third-party datasets or the separately published manuscript. Software tests do not constitute independent scholarly validation.
