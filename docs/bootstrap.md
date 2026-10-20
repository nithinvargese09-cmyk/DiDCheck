# Centered bootstrap

The bootstrap combination is the sum of the mean and variance Wald statistics, $T=W_m+W_q$. It uses a common resampled dataset for both components and therefore retains their dependence. It is not Fisher's statistic and is not a full joint Wald statistic.

## Resampling and centering

For independent units, sample complete unit trajectories with replacement separately within control and treated groups. Keep the selected pretreatment window intact. Never resample period rows independently.

For `cluster_design="stratified_equal_size"`, sample entire group-specific clusters separately within each group. For `cluster_design="shared"`, sample from the pooled collection of independent clusters and keep all units from both groups in each selected cluster. Each repeated selection receives a fresh copy identity so separate copies do not collapse during covariance calculation.

In each draw, recompute both restriction estimates and both covariance matrices. Use centered roots:

$$
T^*=(\widehat b^*-\widehat b)'(\widehat V_b^*)^{-1}
       (\widehat b^*-\widehat b)
    +(\widehat d^*-\widehat d)'(\widehat V_d^*)^{-1}
       (\widehat d^*-\widehat d).
$$

The observed restriction estimates are subtracted before calculating the bootstrap quadratic forms. Ordinary pairs resampling followed by an uncentered Wald statistic generally does not simulate the null law. A larger number of draws or a plus-one denominator cannot repair that problem.

Component bootstrap statistics use the corresponding centered term. The analytic component p-values and bootstrap component p-values are distinct outputs and must be labelled accordingly.

## Monte Carlo convention

For $B$ scheduled draws and $m$ centered statistics at least as large as the observed statistic,

$$
\widehat p=\frac{1+m}{B+1}.
$$

Rejection uses `pvalue <= alpha`. Ties count as exceedances. With 199 draws the smallest possible p-value is 0.005, and a 0.05-level decision rejects when at most nine draws exceed or equal the observed statistic. With 999 draws it rejects when at most 49 do so. The strict rule `pvalue < alpha` would be a different procedure.

The plus-one convention avoids zero Monte Carlo p-values. Exact simulated-rank arguments require exchangeability; the ordinary pairs bootstrap here is justified asymptotically. Increasing $B$ reduces simulation discreteness and conditional Monte Carlo noise, not sampling uncertainty or approximation error. Fix and record the seed, draw count, package version, NumPy version, and data preparation for reproducibility.

## Failed draws

Retain every scheduled draw and its validity status. An invalid required draw makes the corresponding bootstrap calibration unavailable. The implementation does not discard, replace, or keep retrying singular or zero-variance draws. A shared-cluster draw can also be unusable if it loses one group or leaves too few observations for the required calculation.

Report requested, valid, and failed draw counts. Do not interpret an unavailable result as a nonrejection. Diagnose the sample and covariance structure; increasing draws solely until a preferred result appears is not a remedy.

## What the bootstrap requires

Validity requires consistency for the joint centered root, suitable covariance estimation, and a continuous limiting null law. Independent-unit resampling needs independent trajectories; clustered resampling needs independent blocks and appropriate cluster regularity. A supplied cluster label or a large nominal cluster count does not establish these conditions.

The manuscript evaluates selected independent normal, independent centered exponential, serial-normal, and equal-size group-specific Gaussian cluster designs. Those checks do not validate every shared-cluster, unequal-size, overlapping-pair, few-cluster, or heavy-tail application. See [assumptions](assumptions.md) and [simulations](simulations.md).
