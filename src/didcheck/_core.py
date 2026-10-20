"""Newly reconstructed diagnostic; not recovered original analysis.

Whole unit trajectories are independent within and across groups for analyze_batch.
K is fixed; analytic Wald calibration requires finite fourth moments, positive
period variances and nonsingular contrast covariance. Fisher calibration must be
affirmatively enabled only for a justified joint independence argument.
"""
from __future__ import annotations
import math
import numpy as np

VERSION = "1.0.0"
RANK_RTOL = 1e-10


def contrast_matrix(k, reference=0):
    if k < 2 or not 0 <= reference < k:
        raise ValueError("Require K>=2 and a valid reference period")
    rows = [j for j in range(k) if j != reference]
    return np.eye(k)[rows] - np.eye(k)[reference]


def _gamma_q(a, x):
    """Regularized upper incomplete gamma, scalar positive shape.

    Power series for P when x<a+1; continued fraction for Q otherwise.
    Iteration failure raises explicitly. Only used outside optimized df=3,4.
    """
    if math.isnan(x):
        return math.nan
    if x <= 0:
        return 1.0
    if math.isinf(x):
        return 0.0
    logterm = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * 2e-15:
                return min(1.0, max(0.0, 1.0 - total * math.exp(logterm)))
    else:
        tiny = 1e-300
        b = x + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 10000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 2e-15:
                return min(1.0, max(0.0, math.exp(logterm) * h))
    raise ArithmeticError("Incomplete-gamma iteration did not converge")


def chi2_sf(w, df):
    """Chi-square survival; no SciPy dependency. NaN propagates."""
    arr = np.asarray(w, dtype=float)
    if df <= 0:
        raise ValueError("Positive degrees of freedom required")
    out = np.full(arr.shape, np.nan)
    good = ~np.isnan(arr)
    v = np.maximum(arr[good], 0.0)
    if df == 3:
        z = np.sqrt(v / 2)
        erfc = np.fromiter((math.erfc(float(x)) for x in z.flat), float, v.size).reshape(v.shape)
        with np.errstate(invalid="ignore"):
            extra = 2.0 / np.sqrt(np.pi) * z * np.exp(-v / 2)
        extra[np.isinf(v)] = 0.0
        out[good] = erfc + extra
    elif df == 4:
        with np.errstate(invalid="ignore"):
            val = np.exp(-v / 2) * (1 + v / 2)
        val[np.isinf(v)] = 0.0
        out[good] = val
    else:
        out[good] = np.fromiter((_gamma_q(df / 2, float(x) / 2) for x in v.flat), float, v.size).reshape(v.shape)
    return np.clip(out, 0.0, 1.0)


def wald_batch(h, covariance, rtol=RANK_RTOL):
    """Quadratic forms; failed finite/rank checks return NaN and False.

    Never silently use a pseudoinverse or change degrees of freedom.
    """
    h = np.asarray(h, dtype=float)
    v = np.asarray(covariance, dtype=float)
    if h.ndim != 2 or v.shape != (h.shape[0], h.shape[1], h.shape[1]):
        raise ValueError("Expected h=(B,p), covariance=(B,p,p)")
    finite = np.isfinite(h).all(axis=1) & np.isfinite(v).all(axis=(1, 2))
    valid = np.zeros(len(h), dtype=bool)
    stat = np.full(len(h), np.nan)
    idx = np.flatnonzero(finite)
    if idx.size:
        vv = (v[idx] + v[idx].swapaxes(-1, -2)) / 2
        ev = np.linalg.eigvalsh(vv)
        ok = (ev[:, -1] > 0) & (ev[:, 0] > rtol * ev[:, -1])
        chosen = idx[ok]
        if chosen.size:
            sol = np.linalg.solve(vv[ok], h[chosen, :, None])[..., 0]
            values = np.einsum("bi,bi->b", h[chosen], sol)
            valid[chosen] = np.isfinite(values) & (values >= -1e-9)
            stat[chosen] = np.where(valid[chosen], np.maximum(values, 0), np.nan)
    return stat, valid


def _group_moments(y, cluster_size=None):
    n = y.shape[1]
    z = y - y.mean(axis=2, keepdims=True)
    mu = z.mean(axis=1)
    x = z - mu[:, None, :]
    q = np.mean(x * x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        psi = x * x / q[:, None, :] - 1.0
    mean_cov = np.einsum("bni,bnj->bij", x, x) / (n * (n - 1))
    var_cov = np.einsum("bni,bnj->bij", psi, psi) / n**2
    if cluster_size is not None:
        if not isinstance(cluster_size, (int, np.integer)) or cluster_size < 1 or n % cluster_size:
            raise ValueError("cluster_size must be a positive integer dividing group size")
        g = n // cluster_size
        if g < 2:
            raise ValueError("At least two independent clusters per group required")
        sx = x.reshape(len(y), g, cluster_size, y.shape[-1]).sum(axis=2)
        sp = psi.reshape(len(y), g, cluster_size, y.shape[-1]).sum(axis=2)
        correction = g / (g - 1)
        mean_cov = correction * np.einsum("bgi,bgj->bij", sx, sx) / n**2
        var_cov = correction * np.einsum("bgi,bgj->bij", sp, sp) / n**2
    return mu, q, mean_cov, var_cov, x, psi


def _finish(hm, hv, vm, vv, ratios, rho, positive_q, finite_input, k, fisher_valid):
    wm, validm = wald_batch(hm, vm)
    wv, validv = wald_batch(hv, vv)
    validm &= finite_input
    validv &= finite_input & positive_q & (k >= 3)
    wm[~validm] = np.nan
    wv[~validv] = np.nan
    pm, pv = chi2_sf(wm, k - 1), chi2_sf(wv, k - 1)
    enabled = np.broadcast_to(np.asarray(fisher_valid, dtype=bool), validm.shape)
    validf = validm & validv & enabled
    sf = np.full(wm.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf[validf] = -2 * (np.log(pm[validf]) + np.log(pv[validf]))
    pf = chi2_sf(sf, 4)
    return dict(W_mean=wm, W_var=wv, W_sum=wm + wv, S_fisher=sf,
                p_mean=pm, p_var=pv, p_fisher=pf,
                valid_mean=validm, valid_var=validv, valid_fisher=validf,
                finite_input=finite_input, positive_variances=positive_q,
                two_period_degeneracy=np.full(wm.shape, k == 2, dtype=bool),
                h_mean=hm, h_var=hv, cov_mean=vm, cov_var=vv,
                ratios=ratios, log_ratios=rho)


def analyze_batch(y0, y1, reference=0, fisher_valid=False, cluster_size=None):
    """Analyze B independent panels. Inputs (B,N_g,K); dict of batch arrays.

    h_mean=A(mean(Z1)-mean(Z0)); h_var=A(log(s1²)-log(s0²)).
    Sample variances use n-1. Influence vectors use divisor n; its log-ratio
    difference is period-constant and cancels under A. Mean covariance uses
    unbiased trajectory covariance / n; squared-IF covariance uses empirical
    second moments / n. Nonfinite or singular replications are flagged, not
    deleted, replaced or assigned p=1. Caller must report all failure counts.
    Optional cluster_size is an integer or (size0,size1) for equal-size,
    contiguous, group-specific independent clusters. Their summed influence
    vectors replace unit products, with G_g/(G_g-1) finite-cluster correction.
    Shared clusters require analyze_clustered instead.
    """
    y0, y1 = np.asarray(y0, dtype=float), np.asarray(y1, dtype=float)
    if y0.ndim != 3 or y1.ndim != 3 or y0.shape[0] != y1.shape[0] or y0.shape[2] != y1.shape[2]:
        raise ValueError("y0,y1 must have shape (B,N_g,K), same B and K")
    if min(y0.shape[1], y1.shape[1]) < 2:
        raise ValueError("At least two units per group required")
    k = y0.shape[2]
    a = contrast_matrix(k, reference)
    finite = np.isfinite(y0).all(axis=(1, 2)) & np.isfinite(y1).all(axis=(1, 2))
    sizes = (cluster_size, cluster_size) if not isinstance(cluster_size, (tuple, list)) else cluster_size
    if len(sizes) != 2:
        raise ValueError("cluster_size tuple must have two entries")
    m0, q0, cm0, cv0, _, _ = _group_moments(y0, sizes[0])
    m1, q1, cm1, cv1, _, _ = _group_moments(y1, sizes[1])
    s0 = q0 * y0.shape[1] / (y0.shape[1] - 1)
    s1 = q1 * y1.shape[1] / (y1.shape[1] - 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rho = np.log(s1) - np.log(s0)
        ratios = s1 / s0
    hm, hv = (m1 - m0) @ a.T, rho @ a.T
    vm = np.einsum("ij,bjk,lk->bil", a, cm1 + cm0, a)
    vv = np.einsum("ij,bjk,lk->bil", a, cv1 + cv0, a)
    pos = (q0 > 0).all(axis=1) & (q1 > 0).all(axis=1)
    return _finish(hm, hv, vm, vv, ratios, rho, pos, finite, k, fisher_valid)


def analyze(y0, y1, reference=0, fisher_valid=False, cluster_size=None):
    """Single-panel wrapper, inputs (N_g,K), scalar/vector/matrix outputs."""
    res = analyze_batch(np.asarray(y0)[None], np.asarray(y1)[None], reference, fisher_valid, cluster_size)
    return {name: value[0] for name, value in res.items()}


def centered_sum_batch(resampled, original):
    """Original sum of component Wald statistics, with null-centered roots.

    The original can be singleton analyze() output or matching batch arrays.
    No mean-variance cross block is inverted; dependence is retained by using
    the same resampled unit/cluster trajectories for both components.
    """
    wm, vm = wald_batch(resampled["h_mean"] - original["h_mean"], resampled["cov_mean"])
    wv, vv = wald_batch(resampled["h_var"] - original["h_var"], resampled["cov_var"])
    valid = vm & vv & resampled["valid_mean"] & resampled["valid_var"]
    total = wm + wv
    total[~valid] = np.nan
    return total, valid


def _cluster_labels(cluster0, cluster1, n0, n1):
    c0, c1 = list(cluster0), list(cluster1)
    if len(c0) != n0 or len(c1) != n1:
        raise ValueError("Each unit needs one cluster label")
    labels = list(dict.fromkeys(c0 + c1))
    if len(labels) < 2:
        raise ValueError("At least two independent clusters required")
    return c0, c1, labels


def analyze_clustered(y0, y1, cluster0, cluster1, reference=0, fisher_valid=False):
    """Singleton arbitrary/shared clusters, full cluster-summed IF covariance.

    Equal labels in both groups mean the SAME cluster. Sampling validity needs
    many independent clusters, no dominant cluster and a cluster CLT; numerical
    validity flags check finite inputs/rank only. G/(G-1) is a new documented
    finite-cluster convention, not a recovered choice or exact small-G law.
    """
    y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
    if y0.ndim != 2 or y1.ndim != 2 or y0.shape[1] != y1.shape[1] or min(len(y0), len(y1)) < 2:
        raise ValueError("Expected complete (N_g,K) arrays with at least two units per group")
    c0, c1, labels = _cluster_labels(cluster0, cluster1, len(y0), len(y1))
    a = contrast_matrix(y0.shape[1], reference)
    base = analyze(y0, y1, reference, fisher_valid=False)
    _, _, _, _, x0, psi0 = _group_moments(y0[None])
    _, _, _, _, x1, psi1 = _group_moments(y1[None])
    sm = np.zeros((len(labels), y0.shape[1] - 1))
    sv = np.zeros_like(sm)
    indices = {label: j for j, label in enumerate(labels)}
    for sign, c, x, psi, n in [(-1, c0, x0[0], psi0[0], len(y0)), (1, c1, x1[0], psi1[0], len(y1))]:
        for i, label in enumerate(c):
            j = indices[label]
            sm[j] += sign * (a @ x[i]) / n
            sv[j] += sign * (a @ psi[i]) / n
    corr = len(labels) / (len(labels) - 1)
    vm, vv = corr * (sm.T @ sm), corr * (sv.T @ sv)
    out = _finish(base['h_mean'][None], base['h_var'][None], vm[None], vv[None],
                  base['ratios'][None], base['log_ratios'][None],
                  np.array([base['positive_variances']]), np.array([base['finite_input']]),
                  y0.shape[1], fisher_valid)
    result = {name: value[0] for name, value in out.items()}
    result.update(cluster_scores_mean=sm, cluster_scores_var=sv,
                  cluster_labels=labels, n_clusters=len(labels))
    return result
