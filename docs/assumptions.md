# Assumptions and scope

DiDCheck checks data structure and numerical conditions. It cannot verify a population moment condition, sampling independence, or a causal identification assumption from one dataset. A result with `valid=True` is not a certification of the research design.

## Define the sample before testing

- Use two groups and a common, explicitly listed pretreatment window with at least three distinct periods.
- The treatment column identifies group membership and must be constant within each unit. It is not a treatment-on switch that changes after adoption.
- Each included unit must have exactly one observation in every selected period. Duplicate unit-period rows require resolution, not automatic averaging.
- Use finite numeric outcomes. The default missing-data policy raises an error; `missing="drop_units"` explicitly selects complete units over the common window.
- Decide the outcome transformation and sample before inspecting rejections. Changing the scale, window, or retained sample changes the diagnostic being interpreted.

Complete-case selection does not solve selective attrition. When dropping incomplete units, inference concerns the retained sample under a justified sampling model. The package does not model missingness or automatically correct selection bias.

Calling the observations untreated outcomes requires no anticipation in the chosen window. The software cannot infer treatment timing or rule out anticipation. A detected departure does not identify anticipation as its cause.

## Independent-unit analytic inference

The manuscript's benchmark has iid complete unit trajectories within each group and independent sampling across the groups. Periods within a trajectory may be dependent. The asymptotic argument fixes the number of periods while group sizes increase with positive limiting group shares. It requires finite fourth outcome moments, positive transformed period variances, and positive-definite limiting contrast covariance.

The numerical eigenvalue rule detects singular or nearly singular estimated covariance, not whether the population covariance or asymptotic approximation is appropriate. Two units per group may satisfy a minimum input rule but cannot be treated as a generally adequate sample size. There is no universal minimum sample-size threshold for reliable inference or useful power.

## Independent clusters

Cluster labels must identify the actual independent sampling blocks. Dependence may occur within a block and, for shared clusters, across both groups in the same block. Valid inference needs many independent clusters, a cluster-level central limit theorem, consistent score covariance, and no dominant cluster.

One sufficient benchmark is iid independent cluster blocks with uniformly bounded sizes, finite fourth outcome moments, positive limiting group shares, and nonsingular limiting component covariance. More general cluster arrays require their own regularity argument. Finite fourth unit-level moments alone do not cover unrestricted cluster sizes.

The API requires an explicit design:

| `cluster_design` | Meaning |
| --- | --- |
| `"stratified_equal_size"` | Group-specific clusters, with equal cluster size within each group; resample whole clusters separately within groups. |
| `"shared"` | A pooled collection of independent clusters; matching labels across groups denote the same cluster and both groups are resampled together. |

These options are not interchangeable finite-sample conventions. A row-order block or a cluster label does not itself establish independence. Overlapping matched pairs, spatial spillovers, and common shocks may require larger blocks or a different inferential design. The package does not infer those blocks from the data.

## Fisher requires a separate justification

Fisher's usual chi-square calibration requires jointly asymptotically independent component p-values under their joint null, as well as correctly calibrated components. In the independent-unit benchmark, central symmetry of the whole centered trajectory makes the relevant mean/variance cross-third moments zero. Joint normal limits and consistent studentization then supply asymptotic independence.

Whole-trajectory symmetry is sufficient, not necessary; vanishing relevant cross-third moments can also suffice. Marginal symmetry, individually cluster-robust p-values, or visually symmetric histograms do not establish the required joint condition. For clustering, the argument must apply to the actual cluster-level scores. The package records a supplied justification but cannot verify it.

A classical spherical-Gaussian special case can give exact independence for particular statistics and scalar-pooled mean studentization. The implemented full-covariance mean Wald statistic uses asymptotic calibration; that special case does not make its p-values exact. Use the centered bootstrap sum when its assumptions are justified and Fisher independence is not established.

## Scope boundaries

This implementation does not supply survey weights, automatic covariate adjustment, a model for incomplete trajectories, treatment-effect estimation, or an automatic staggered-adoption/event-stacking procedure. It does not establish validity for arbitrary few-cluster, shared-cluster, unequal-cluster, heavy-tail, or overlapping-pair designs merely because it can compute a number.

The manuscript's empirical applications remain unvalidated. Its simulation results describe the specified synthetic designs and do not establish that the China Shock, minimum-wage, or another empirical setting satisfies these assumptions.
