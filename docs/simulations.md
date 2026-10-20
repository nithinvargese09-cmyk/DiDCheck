# Synthetic data and simulations

The examples and generator produce synthetic panels. They do not reconstruct an empirical China Shock or minimum-wage sample. Running one panel demonstrates the API; it does not estimate size or power.

## Panel generator

```python
from didcheck import simulate_panel

simulate_panel(
    n_control=200, n_treated=200, n_periods=4,
    mean_delta=0.0, variance_delta=0.0, law="normal",
    phi=0.0, cluster_size=1, cluster_rho=0.0,
    random_state=0,
)
```

The result is a long-format pandas DataFrame with `unit`, `time`, `treated`, `outcome`, and `cluster` columns. Periods are the integers 1 through `n_periods`; treatment is constant group membership. Unit and cluster IDs are distinct across groups. The simulation settings are also stored in `frame.attrs["simulation"]`.

The model is

$$
Y_{git}=\alpha_{gi}+\tau_t+\mathbf1\{g=1\}\delta_m t+\varepsilon_{git},
\qquad \alpha_{gi}\sim N(0,0.25),\quad\tau_t\sim N(0,0.09).
$$

Each time effect is common to both groups. The diagnostic conditions on these realized common effects, which cancel from its restrictions. Unit effects, time effects, and innovations are generated independently. Control error variance is one; treated errors receive the multiplier $\sqrt{1+\delta_vt}$, giving original disturbance variance $1+\delta_vt$. This is an additive variance slope, not compounded percentage growth. It is not itself the demeaned variance ratio.

| `law` | Standardized innovation |
| --- | --- |
| `"normal"` | Standard normal |
| `"uniform"` | Uniform on $[-\sqrt3,\sqrt3]$ |
| `"t5"` | Student $t(5)$ multiplied by $\sqrt{3/5}$ |
| `"laplace"` | Laplace with scale $1/\sqrt2$ |
| `"exponential"` | Mean-one exponential minus one |

All five have mean zero, variance one, and finite fourth moments. The centered exponential law is asymmetric; the package does not endorse analytic Fisher calibration for that design.

`phi` supplies Gaussian AR(1) correlation $\phi^{|t-s|}$, with $-1<\phi<1$. `cluster_size>1` creates group-specific equal-size clusters and requires normal innovations. Cluster errors combine a common cluster trajectory and an idiosyncratic trajectory with variance shares `cluster_rho` and `1-cluster_rho`; both use the same AR setting. The treated variance multiplier is applied afterward. `cluster_rho` lies in $[0,1]$, and a nonzero share requires `cluster_size>1`. Cluster size must divide both group sizes. The generator does not create clusters shared across groups.

Both groups need at least two units, and all treated period variances must stay positive. The generator allows two periods to support explicit degeneracy illustrations, but the public diagnostic requires at least three. Group sizes can differ and period counts can vary; these are software extensions beyond the manuscript's four-period equal-group simulation grid.

## Runnable examples

From the repository root after installation:

```sh
python examples/synthetic_panel.py
python examples/clustered_panel.py
```

Both scripts use fixed seeds and 199 bootstrap draws by default, print a labelled synthetic result, and leave Fisher disabled. `--bootstrap 0` requests analytic components only. `--seed` changes the synthetic sample. `--plot` displays descriptive figures when Matplotlib is installed. The clustered example has 40 separate clusters of five units per group, with `phi=0.5` and `cluster_rho=0.4`, and explicitly selects stratified equal-size cluster inference.

Do not keep changing seeds until the example rejects. Its purpose is to show inputs and outputs, not to promise a particular decision.

## Bounded Monte Carlo helper

The optional analytic helper is available from the simulation module:

```python
from didcheck.simulation import monte_carlo

study = monte_carlo(
    repetitions=100,
    n_control=200, n_treated=200, n_periods=4,
    mean_delta=0.0, variance_delta=0.0,
    law="normal", phi=0.0, cluster_size=1, cluster_rho=0.0,
    random_state=123, batch_size=100,
    reference=0, alpha=0.05, fisher_valid=False,
)
print(study["summary"])
```

This small run illustrates the helper; 100 panels do not provide precise calibration evidence. The signature includes the generator settings plus `repetitions`, `batch_size`, zero-based `reference`, `alpha`, and `fisher_valid`. Fisher opt-in is restricted to the generator's symmetric laws and relies on its stated symmetry argument. It remains asymptotic. This helper evaluates analytic tests; it does not launch a bootstrap calibration study or the manuscript's full historical grid.

The returned dictionary contains:

- `summary`: one row each for mean, variance, and Fisher, with enabled status, scheduled/valid/failed/disabled counts, rejections, rejection frequency, Monte Carlo standard error, and pointwise Wilson interval endpoints.
- `raw`: all scheduled statistics, component p-values, numerical-validity flags, and rejection flags.
- `configuration`: the complete settings needed to describe the run.

Rates and interval endpoints are fractions, not percentages. Rejection uses `pvalue <= alpha`. Every scheduled replication stays in the denominator; no numerical failure is replaced. When failures occur, the all-scheduled rejection frequency describes what the procedure recorded and must be reported alongside failure counts. It is not a calibrated size or power claim. Disabled Fisher is recorded separately from numerical failures.

Batching limits memory used for generated panels, while saved raw output grows with the number of replications. Hold the seed, batch size, settings, and software environment fixed for replay. Identical seeds across different batches or versions need not reproduce the same panels.

## Relationship to the manuscript's evidence

The manuscript reports a separately specified study: 198,000 principal analytic panels, eight selected bootstrap scenarios with 1,000 outer panels and 199 inner draws each, and a separately reported post-principal small-sample check. These are historical manuscript results, not counts of tests or simulations newly executed by installing DiDCheck. The four-period equal-group generator preserves the documented research construction, but a single example or short helper run is not a reproduction of the complete study.

That evidence includes analytic small-sample overrejection, limited power against modest variance departures, and selected improvements from centered bootstrap calibration. It does not support universal superiority of a combined test or validation of arbitrary empirical or cluster designs. The separate manuscript reproduction package preserves the historical configurations and raw outputs. Use the current repository's executed validation report for what was actually checked for this software version; do not transfer historical check counts to the new interface.
