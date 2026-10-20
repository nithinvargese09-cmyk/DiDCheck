# DiDCheck 0.1.0 validation report

Prepared 2026-10-18; local release checks repeated 2026-10-19. This report describes executed local software checks and partial reproduction of the finalized October 2026 paper. It does not claim external review, a complete Monte Carlo rerun, empirical validation, or publication.

## Executed results

| Check | Executed outcome |
| --- | --- |
| Source-level automated suite | 120 tests and 65 subtests passed; zero final failures |
| Fresh installed-wheel suite | 120 tests and 65 subtests passed; zero final failures |
| Isolated import from installed distribution | Passed; version 0.1.0 |
| Dependency consistency (`pip check`) | No broken requirements |
| Source distribution and wheel build | Passed; both passed `twine check` |
| Original bounded replay | 102/102 comparisons bit-exact and within declared tolerances |
| Three complete original analytic cases | 15,000 panels; 24/24 arrays bit-exact; all nine test rejection counts agree |
| Synthetic independent-unit example | Passed with 199 scheduled resamples |
| Synthetic group-specific cluster example | Passed with 199 scheduled resamples |
| Plotting | Finite/failed-draw cases tested; two synthetic preview plots visually inspected |
| Frozen core preservation | Byte-for-byte match to the finalized scientific implementation |
| Public package screening | No local installation/user paths or credential patterns found in the screened public files; only numeric simulated reference arrays included |

Machine-readable records are in `validation/clean_install.json`, `reference_validation.json`, `clean_install_reference.json`, `paper_benchmarks.json`, and `clean_install_benchmarks.json`. Recorded environments and hashes distinguish what was actually tested from supported-but-unexecuted configurations.

## Mathematical and software checks

Independent explicit-loop calculations check within-unit transformations, mean contrasts, squared-influence cross-fourth moments, full temporal covariance, the `n(n-1)` mean-covariance and `n^2` variance-covariance divisors, and Wald quadratic forms. Finite-support calculations check `M Omega M` and `AM=A`; tests check the Gaussian covariance benchmark, all reference periods, group rescaling, unit shifts and separate mean/variance departures.

Bootstrap checks independently reconstruct centered roots and resampled covariance estimates. They verify the sum of component Walds, shared resamples, the inclusive plus-one tail count, ties, scheduled draw retention and failure propagation. Cluster checks cover signed shared-cluster scores, both finite-cluster conventions, whole-block draws, duplicate-copy identities and varying group denominators.

Input/integration tests cover reordered periods, row shuffling, array/DataFrame agreement, stable binary treatment membership, duplicate observations, missing complete trajectories, explicit whole-unit dropping, cluster metadata, nonfinite and complex outcomes, two-period degeneracy in the low-level core, singular/zero-variance cases, invalid parameter types, fixed seeds, partial batches, result warnings and plotting. Cross-layout floating-point comparisons use tight tolerances; deterministic row-shuffle checks retain exact comparisons. Core chi-square tails are checked against SciPy and independent numerical calculations in the retained validation history.

## Reproduced paper findings

The three cases use their original stage-1 PCG64 `SeedSequence` streams, batch size 100, four periods, 5,000 panels per case, and the unchanged numerical core. Full expected statistic/p-value/validity arrays are compactly bundled with provenance.

| Original case | Mean rejection | Variance rejection | Fisher rejection |
| --- | ---: | ---: | ---: |
| Gaussian null, 50 units/group | 6.04% (302) | **8.44% (422)** | 8.00% (400) |
| Gaussian null, 2,000 units/group | 4.94% (247) | **4.96% (248)** | 5.02% (251) |
| Variance alternative 0.08, 2,000 units/group | 5.12% (256) | **49.02% (2,451)** | 38.40% (1,920) |

All 15,000 panels were numerically valid. All 24 complete saved arrays matched bit-for-bit, with maximum absolute error zero in the recorded environment. This is same-core reproduction, distinct from the independent formula tests. The bounded prefix check additionally covers 300 analytic panels and the first outer panel plus 199 inner draws for each of eight original bootstrap cases.

These results preserve the adverse small-sample calibration and limited-power findings. They do not support uniform superiority of combined tests, exact finite-sample calibration, or validation of a causal design.

## Problems encountered and resolved

The integration review identified parameter-validation issues: malformed alpha and cluster containers leaked type errors, scalar strings could be read as sequences of cluster labels, and complex inputs could silently lose imaginary components. Public validation now rejects these inputs explicitly, with regression tests. A newly added Monte Carlo reporting helper had floating-point Wilson endpoints near zero/one; its endpoints now use exact boundary values. These changes do not alter the frozen scientific core or published numbers.

Initial cross-layout integration assertions demanded bit equality for mathematically identical array layouts. Differences were at floating-point rounding scale (about 1e-16); the checks now use tight numerical tolerances for that comparison while keeping discrete p-values and seeded row-shuffle replay exact. An editable-install path was not loaded by the development interpreter; ordinary wheel installation resolved that local environment issue. Final verification uses a separate clean environment and the installed wheel, not editable-path injection.

No unresolved failing tests or discrepancies requiring a paper correction were found in the executed scope. The original finalized paper and saved study outputs were not modified.

## Environments and unexecuted scope

The local scientific environment uses Python 3.12.3, NumPy 2.4.3, SciPy 1.18.1, pandas 3.0.3, Matplotlib 3.10.8 and pytest 9.0.3 on macOS/ARM64. The fresh environment was created without system site packages. Direct requirements are pinned in `requirements-validation.txt`; transitive versions are recorded in `validation/clean_install.json`.

The full 198,000-panel analytic grid, all 8,000 principal bootstrap outer panels and the separate 1,000-panel diagnostic were **not rerun for DiDCheck development**. Their complete saved outputs belong to the separate archival replication package. No historical China Shock or minimum-wage p-value was reconstructed or endorsed.

Python >=3.10 is the intended package range; Python 3.12.3/macOS ARM64 was executed locally. [GitHub Actions run 39358833207](https://github.com/nithinvargese09-cmyk/DiDCheck/actions/runs/39358833207) passed on Ubuntu for Python 3.10, 3.11, 3.12, and 3.13, including the automated suite, 102 reference comparisons, 15,000-panel benchmarks, both examples, builds, and metadata checks. Windows, all supported dependency combinations, and other BLAS/platform implementations remain untested. Cross-environment bit equality is not guaranteed; the reference scripts separately report bit and tolerance agreement.

## Remaining methodological limits

The method requires a common complete window and positive variances/nonsingular contrast covariance; the high-level variance API requires at least three periods. The target is transformed dispersion after within-unit demeaning. It is not generally the original disturbance variance, and equality of CDF changes does not imply the diagnostic null.

Asymptotic calibration requires the paper's sampling and moment conditions. Shared/unequal-cluster helpers have computational checks but do not inherit blanket Monte Carlo validation from the selected 40-by-five group-specific cluster study. Bootstrap plus-one counting does not repair an invalid resampling law. Fisher needs a justified joint-independence argument. There is no generic chi-square law for the sum, no simultaneous error-control claim across reported tests, and no treatment-effect or variance-ratio confidence interval implementation.

A real-world application is not supplied because the finalized project does not uniquely specify validated empirical samples and dependence designs. Synthetic examples are explicitly labeled. The companion working paper is available on Zenodo as [Distributional Pre-Trends: A Variance Diagnostic for Difference-in-Differences Designs](https://doi.org/10.5281/zenodo.23328869), published October 13, 2026, by Aditya Alaparti, Amann Katta, Nithin Vargese, and Swaraj Sainani. This verified preprint citation does not imply peer review. This report records local checks; public release status and remote CI results are recorded separately in the GitHub release. A PyPI HTTP-404 name check is not a reservation.
