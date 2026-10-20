# DiDCheck

[![Tests](https://github.com/nithinvargese09-cmyk/DiDCheck/actions/workflows/tests.yml/badge.svg)](https://github.com/nithinvargese09-cmyk/DiDCheck/actions/workflows/tests.yml)

DiDCheck implements the pretreatment moment diagnostics in **Distributional Pre-Trends: A Variance Diagnostic for Difference-in-Differences Designs**, by Aditya Alaparti, Amann Katta, Nithin Vargese, and Swaraj Sainani (2026). All four authors contributed equally. All share the affiliation Thomas Jefferson High School for Science and Technology, Alexandria, Virginia, United States.

The variance diagnostic tests whether the treated/control ratio of variances is constant across a specified common pretreatment window **after within-unit demeaning**. It uses full temporal covariance, offers the companion mean diagnostic, and implements null-centered whole-unit or whole-cluster pairs bootstrap inference for the components and their sum.

This is companion research software, version 0.1.0. It is not a treatment-effect estimator or a certificate of parallel trends. Rejection concerns a stated moment restriction; nonrejection does not establish equivalence or causal identification. Finite-sample calibration and power remain limited in the paper's evaluated settings.

**Research preprint:** [Distributional Pre-Trends](https://doi.org/10.5281/zenodo.23328869), Zenodo working paper, October 13, 2026. This citation does not imply peer review.

## Installation

For the published release:

```sh
python -m pip install didcheck==0.1.0
```

### Install and test from source

Python 3.10 or newer is required. From the extracted repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[test]"
python -m pytest
```

On Windows, activate with `.venv\Scripts\activate`. Alternatively install the included wheel with `python -m pip install dist/didcheck-0.1.0-py3-none-any.whl`. Dependencies are NumPy, SciPy, pandas, and Matplotlib. See [release instructions](docs/release.md) for the maintainer workflow.

## Quick start: synthetic data

```python
from didcheck import simulate_panel, variance_test

df = simulate_panel(n_control=200, n_treated=200, n_periods=4,
                    variance_delta=0.20, random_state=42)
result = variance_test(
    df, outcome="outcome", unit="unit", time="time", treatment="treated",
    pre_periods=[1, 2, 3, 4], bootstrap=199, random_state=17,
)
print(result.summary())
print(result.variance.statistic, result.variance.pvalue)
print(result.bootstrap["variance"]["pvalue"])
print(result.combined.pvalue)  # centered bootstrap for W_mean + W_variance

from didcheck.plotting import plot_variance_ratios
ax = plot_variance_ratios(result)
ax.figure.savefig("variance_ratios.png", dpi=150, bbox_inches="tight")
```

This example produces synthetic illustration data, not empirical evidence. The integer `bootstrap` is the scheduled draw count; use a prespecified count appropriate to Monte Carlo precision. The minimum possible bootstrap p-value is `1/(B+1)`.

## Input and inference contract

- Supply an ordered list of at least three **pretreatment** periods. DiDCheck does not infer treatment timing or choose a window.
- `treatment` is a fixed binary group indicator, not a post-treatment switching indicator. Staggered-adoption designs are not implemented.
- One observation per unit and selected period is required. Missing/incomplete trajectories raise by default. `missing="drop_units"` explicitly selects complete cases and reports dropped units; it does not solve selective attrition.
- The default inference assumes independent whole-unit trajectories within independent groups. For clusters, explicitly choose `cluster_design="stratified_equal_size"` or `"shared"` with a cluster column. Their sampling laws and finite-cluster corrections differ.
- Fisher's combination is off by default. `fisher=True` requires a nonempty `fisher_justification` describing joint asymptotic independence; this records an assumption rather than verifying it.
- The combined sum has no generic chi-square calibration. Its p-value requires the centered bootstrap.
- Rank failures, zero variances, and failed bootstrap draws remain visible. No pseudoinverse, reduced-rank substitution, retry, or silent removal of invalid draws is used.
- Multiple reported tests do not constitute a familywise-error-controlled testing procedure. No treatment-effect or variance-ratio confidence bands are provided.

For array input use `diagnose_arrays(control, treated, ...)`, with two arrays shaped `(units, periods)`. See the [API](docs/api.md), [methodology](docs/methodology.md), [assumptions](docs/assumptions.md), and [bootstrap documentation](docs/bootstrap.md).

## Reproduce examples and validation

```sh
python examples/synthetic_panel.py
python examples/clustered_panel.py
python validation/reference_check.py
python validation/reproduce_paper_benchmarks.py
```

The compact frozen fixtures contain selected original simulation prefixes and three complete analytic cases. Reference checking regenerates 300 analytic panels and the first outer panel with its 199 resamples in each of eight bootstrap scenarios. The benchmark command separately reproduces three complete 5,000-panel cases, checking the reported variance rejection rates of 8.44%, 4.96%, and 49.02%. Neither command reruns the entire study. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for executed tests, exact versus tolerance agreement, environments, failures and remaining limits. [Simulation utilities](docs/simulations.md) support declared synthetic scenarios and preserve scheduled/valid/failed counts; example outputs are never represented as published findings.

The paper's full study included 198,000 principal analytic panels, 8,000 principal bootstrap outer panels, and a separate 1,000-panel diagnostic, with 199 inner resamples per bootstrap panel. This software-development validation does not claim a complete rerun. Original finalized manuscripts and saved study outputs are not modified. The China Shock and minimum-wage settings remain prospective applications, not validated empirical results.

## Repository contents

`src/didcheck/` contains the public API, frozen numerical core, covariance utilities, bootstrap, simulation, plotting and result objects. `tests/` contains formula, resampling, data-contract and edge-case tests. `validation/` contains compact frozen references, scripts and provenance. `examples/` and `docs/` provide reproducible usage and methodological guidance. `dist/` in the delivery ZIP contains the built wheel and source distribution.

## Citation, attribution and license

Use [CITATION.cff](CITATION.cff). This project implements the specific methods in the companion paper; it does not claim a new method beyond that contribution. The internal `_core.py` is preserved byte-for-byte from the finalized scientific implementation, with its SHA-256 in `validation/CORE_PROVENANCE.json`. See [attribution and AI assistance](docs/attribution.md).

BSD-3-Clause, selected by the project owner; see [LICENSE](LICENSE). No institutional endorsement is implied. Third-party datasets, private review conversations, historical drafts, credentials and local-machine logs are excluded. The package includes synthetic reference outputs only.
