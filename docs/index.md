# DiDCheck documentation

DiDCheck computes complementary mean and variance diagnostics for two groups observed over a common pretreatment window. Its variance null is a constant treated-to-control variance ratio **after each unit is demeaned over that window**. The groups may have different variance levels.

Start with the [assumptions](assumptions.md) and [API](api.md), then run the [synthetic independent-unit example](../examples/synthetic_panel.py) or [synthetic clustered example](../examples/clustered_panel.py).

| Guide | Purpose |
| --- | --- |
| [Methodology](methodology.md) | Target, estimators, covariance, test statistics, and interpretation |
| [Assumptions](assumptions.md) | Sampling, panel structure, moments, Fisher conditions, and unsupported designs |
| [API](api.md) | Data preparation, arguments, results, and plotting |
| [Bootstrap](bootstrap.md) | Null centering, resampling units, cluster conventions, and failed draws |
| [Simulations](simulations.md) | Synthetic generator, study boundaries, and reproducible evaluation |
| [Troubleshooting](troubleshooting.md) | Input errors, singularity, unavailable results, and interpretation |
| [Release](release.md) | Local release checks and publication status |
| [Attribution](attribution.md) | Manuscript attribution and relationship to the software |

Rejection provides evidence against the specified pretreatment moment restriction. It does not identify the cause, establish bias in a treatment effect, or test every distributional restriction. Nonrejection does not validate parallel trends or demonstrate equivalence. DiDCheck does not estimate treatment effects.

The accompanying examples are synthetic. Neither the China Shock nor the minimum-wage application discussed as future research in the manuscript is a validated DiDCheck application.
