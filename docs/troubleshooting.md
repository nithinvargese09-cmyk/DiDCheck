# Troubleshooting

## Input errors

| Symptom | Interpretation and next step |
| --- | --- |
| Fewer than three selected periods | The demeaned variance test degenerates with two periods. Supply a substantively justified longer common pretreatment window; do not fabricate periods. |
| Duplicate unit-time observations | Identify why multiple records represent the same unit and period. The package does not silently average them. |
| Time-varying treatment column | Supply a stable group indicator. A post-adoption treatment switch is not the required input, and pooled staggered-adoption analysis is unsupported. |
| Incomplete trajectories | Correct data preparation or explicitly choose `missing="drop_units"` for a defensible complete-case analysis. Report dropped IDs and possible selection. |
| Infinite outcomes | Correct the invalid transformation or measurement; dropping missing units does not silently permit infinities. |
| Cluster-design error | Supply an explicit supported design matching independent sampling blocks. Stratified clusters must be group-specific and equal-size within each group. |
| Fisher-justification error | Keep `fisher=False` unless an appropriate joint-independence argument is available. A nonempty sentence records an argument but does not prove it. |

The DataFrame interface validates group and cluster constancy in the supplied data, including observations outside the selected window. This prevents a time-varying treatment switch from being mistaken for a stable group indicator merely because it is constant during the selected preperiods.

## Unavailable or singular results

Zero transformed variances prevent log-variance inference. Identical trajectories, too few independent units or clusters, deterministic relationships, and nearly redundant contrasts may produce singular covariance matrices. The implementation requires the smallest eigenvalue to exceed $10^{-10}$ times the largest and does not silently change rank or degrees of freedom.

Inspect `.warnings`, component `.valid`, group counts, the selected period window, and the full covariance matrices. An unavailable p-value is `NaN`, and its rejection decision is `None`. Treating it as `False` or filling it with one would misreport a failed calculation as a nonrejection.

The sum has no calibrated p-value unless bootstrap is requested. Fisher is unavailable by default. These default statuses are deliberate, not failures of the component calculations.

If a required bootstrap draw is invalid, inspect `result.bootstrap[component]["failed_draws"]` and its validity mask. The failed draws remain in the record and are not replaced. Shared-cluster resampling can lose one group in a draw; small effective cluster counts or nearly singular data can also make resampling unreliable. Do not discard failed draws or retry until a preferred result is obtained.

## Unexpected p-values

A large p-value does not establish equivalent trends or validate a causal design. Limited power can be substantial even with many units. A small variance p-value is evidence about the demeaned variance-ratio restriction, not direct evidence of mean treatment-effect bias.

Analytic and bootstrap p-values can differ because the first uses an asymptotic chi-square reference and the second uses a finite set of centered resamples. More draws reduce Monte Carlo noise but do not repair invalid assumptions. Inspect the ratio path and both components, and report the chosen primary diagnostic and any multiplicity rule.

A nonlinear outcome transformation, a different common window, or complete-case selection can change the target. Changing the reference period should leave a well-conditioned full-covariance Wald statistic unchanged apart from numerical precision; changing the set of periods is a different operation.

## Reproducibility and plots

Record data-preparation code, package and NumPy versions, period order, reference, clustering design, seed, batch size, and bootstrap count. Identical seeds alone do not guarantee identical random draws across different versions or generation paths.

`plot_bootstrap` needs a result with stored draws and a supported component name. Plotting functions require Matplotlib and return an Axes; use `ax.figure.savefig(...)` to save a plot or `matplotlib.pyplot.show()` in an interactive session. The variance-ratio plot is descriptive and has no confidence band.

When reporting an issue, use a minimal synthetic reproducer where possible. Include the full exception, parameters, versions, relevant validity flags, and independent-unit or cluster counts. Do not include confidential research data solely to demonstrate an API issue.
