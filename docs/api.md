# Public API

Import the entry points with:

```python
from didcheck import diagnose_arrays, simulate_panel, variance_test
```

## Long-format panels

```python
variance_test(
    data, *, outcome, unit, time, treatment, pre_periods,
    reference=None, missing="raise", cluster=None, cluster_design=None,
    alpha=0.05, fisher=False, fisher_justification=None,
    bootstrap=0, random_state=0, batch_size=100,
)
```

`data` is a pandas DataFrame. The column-name arguments and `pre_periods` are required; the function does not guess column names or pretreatment dates.

| Argument | Meaning |
| --- | --- |
| `outcome` | Numeric outcome column; transformed by within-unit demeaning over the selected window. |
| `unit` | Unit identifier, constant across its repeated observations. |
| `time` | Period label column. |
| `treatment` | Time-invariant binary 0/1 **group membership**, with 0 control and 1 treated. |
| `pre_periods` | Ordered sequence of at least three unique, nonmissing common pretreatment labels. Supplied order is retained. |
| `reference` | One selected period label; `None` uses the first supplied period. |
| `missing` | `"raise"` or explicit complete-case selection with `"drop_units"`. |
| `cluster` | Optional column identifying each unit's constant cluster membership. |
| `cluster_design` | Required when `cluster` is supplied: `"shared"` or `"stratified_equal_size"`. |
| `alpha` | Finite significance threshold strictly between zero and one; rejection uses `<=`. |
| `fisher` | Boolean, default `False`; enables qualified asymptotic Fisher calibration. |
| `fisher_justification` | Nonempty string required with `fisher=True`; records the user's independence argument, not a software verification. |
| `bootstrap` | Nonnegative integer number of scheduled draws; zero disables bootstrapping. |
| `random_state` | Nonnegative integer seed, or `None` for an unseeded run. |
| `batch_size` | Positive integer batch size; retain it for exact seeded replay. |

The unit, time, treatment, outcome, and optional cluster columns must be distinct. Selected unit-period observations must be unique. Each retained group needs at least two units, although useful inference generally requires much more information and a nonsingular covariance. Rows outside `pre_periods` are excluded from the calculation; identifier and constant-membership checks still apply to the supplied dataset. A unit with no observations in the selected window is incomplete and follows the missing-unit policy.

Infinite outcomes are rejected even with `missing="drop_units"`. The latter policy removes entire incomplete trajectories, not individual period rows, and reports dropped unit IDs in `result.metadata["dropped_units"]`. It does not impute missing outcomes or validate complete-case selection.

```python
panel = simulate_panel(random_state=12)  # Synthetic data only.
result = variance_test(
    panel,
    outcome="outcome", unit="unit", time="time", treatment="treated",
    pre_periods=[1, 2, 3, 4],
    bootstrap=199, random_state=42,
)
print(result.summary())
```

For group-specific equal-size clusters, labels must be disjoint across groups, with at least two clusters per group and equal cluster sizes within each group. The DataFrame interface orders complete unit rows by cluster before calling the array implementation. For `"shared"`, equal labels across groups denote the same cluster. Do not rename genuinely shared clusters to force a stratified design.

## Array interface

```python
diagnose_arrays(
    control, treated, *, reference=0, alpha=0.05,
    fisher=False, fisher_justification=None,
    bootstrap=0, random_state=0, batch_size=100,
    cluster_size=None, clusters=None,
)
```

Each input has shape `(units, periods)` with finite values, at least two unit rows, and the same number of period columns, at least three. Columns must represent the same ordered pretreatment periods; no date or group validation is possible from these arrays alone. `reference` is a zero-based column index. Array results use period labels `0, ..., K-1`.

- With neither clustering argument, the sampling assumption is independent complete units within independent groups.
- `cluster_size=m` uses consecutive blocks of `m` unit rows within each group. A pair `(m_control, m_treated)` permits a different fixed size in each group. Each size must divide its group count, with at least two clusters per group. This selects the stratified equal-size convention.
- `clusters=(control_labels, treated_labels)` supplies one nonmissing scalar label per row and selects pooled whole-cluster inference. Matching labels across groups identify the same block.
- `cluster_size` and `clusters` are mutually exclusive. Array row order and labels are the caller's responsibility.

Explicit `cluster_size=1` still selects the finite-cluster covariance convention. It is not numerically identical to `cluster_size=None`, which selects the independent-unit convention. Similarly, passing the generator's `cluster` column when every cluster has one unit is an explicit sampling-convention choice, not a required step for independent-unit analysis.

## Results

Both functions return `DiagnosticResult`. Its `.mean`, `.variance`, `.fisher`, and `.combined` attributes are `TestResult` objects with:

| Field | Meaning |
| --- | --- |
| `statistic` | Observed statistic; may be unavailable if numerical conditions fail. |
| `pvalue` | Calibrated p-value, or `NaN` when unavailable or disabled. |
| `df` | `K-1` for analytic components, 4 for Fisher, and `None` for the bootstrap sum. |
| `reject` | Boolean for an available p-value, otherwise `None`. |
| `valid` | Computational availability under the selected calibration, not proof of assumptions. |
| `calibration` | Description of the reference distribution or resampling method. |

`.mean` and `.variance` always refer to the analytic component calibrations. `.fisher` remains disabled unless requested with a justification. `.combined` is the centered-bootstrap calibration of the sum. Without bootstrap, it contains the observed sum but no calibrated p-value, with `valid=False` and `reject=None`. No unsupported chi-square degrees of freedom are assigned to it.

Additional fields include:

- `.variance_ratios`, `.log_variance_ratios`: one value per selected period, from demeaned outcomes.
- `.mean_contrast`, `.variance_contrast`: `K-1` nonreference-minus-reference contrasts, in selected-period order with the reference omitted.
- `.mean_covariance`, `.variance_covariance`: full contrast covariance matrices.
- `.periods`, `.reference`, `.n_control`, `.n_treated`, and `.alpha`: analysis details.
- `.warnings`: explanatory messages, including disabled calibration, failures, or complete-case selection.
- `.metadata`: sampling and Fisher justification; the DataFrame interface also records unit IDs, dropped units, and missing-data policy.
- `.summary()`: human-readable report.
- `.to_frame()`: a pandas table of analytic tests and the sum, plus component bootstrap rows when requested. The bootstrap sum appears once as `combined`.

## Bootstrap result dictionary

`.bootstrap` is `None` when no draws were requested. Otherwise, top-level keys include `repetitions`, `random_state`, `resampling`, `batch_size`, and component entries `mean`, `variance`, and `sum`. Each component entry contains:

| Key | Meaning |
| --- | --- |
| `statistic` | Corresponding observed statistic. |
| `pvalue` | Plus-one exceedance p-value when calibration is available. |
| `statistics` | Array of scheduled centered bootstrap statistics. |
| `valid_draws` | Boolean mask with one validity flag per scheduled draw. |
| `failed_draws` | Count of invalid required draws. |
| `exceedances` | Exceedance count when available; inspect `valid` before interpreting it. |
| `valid` | Whether that bootstrap calibration is available. |

For example, `result.bootstrap["variance"]["pvalue"]` is the bootstrap variance p-value, distinct from `result.variance.pvalue`. Read [bootstrap conventions](bootstrap.md) before interpreting finite-draw results.

## Plotting

```python
from didcheck.plotting import plot_bootstrap, plot_variance_ratios

ratio_ax = plot_variance_ratios(result)
bootstrap_ax = plot_bootstrap(result, component="sum")
```

Both functions accept `ax=None` or an existing Matplotlib Axes and return the Axes. Matplotlib is needed only when plotting. `plot_variance_ratios` draws the descriptive demeaned-variance ratio path without confidence intervals. `plot_bootstrap` accepts `"mean"`, `"variance"`, or `"sum"`; it requires stored bootstrap draws and displays the observed statistic and failed-draw count. Displaying finite draws does not make an otherwise invalid calibration valid. Neither plot estimates a treatment effect or provides treatment-effect confidence intervals.

See [simulations](simulations.md) for `simulate_panel` and the runnable synthetic examples.
