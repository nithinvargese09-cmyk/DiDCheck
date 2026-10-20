"""Reproducible panel generators and a bounded-memory Monte Carlo helper.

The four-period equal-group generator preserves the finalized study's draw
order and DGP. General period counts and unequal group sizes are extensions;
small runs of this module do not reproduce the full published simulation study.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import _core
from .bootstrap import _positive_integer, _random_generator

K = 4
_LAWS = ("normal", "uniform", "t5", "laplace", "exponential")


def standard_errors(rng, shape, law):
    """Draw the study's mean-zero, unit-variance innovations."""
    if law == 'normal': return rng.standard_normal(shape)
    if law == 'uniform': return rng.uniform(-np.sqrt(3), np.sqrt(3), shape)
    if law == 't5': return rng.standard_t(5, shape) * np.sqrt(3/5)
    if law == 'laplace': return rng.laplace(0, 1/np.sqrt(2), shape)
    if law == 'exponential': return rng.exponential(1, shape)-1
    raise ValueError(law)


def ar_transform(x, phi):
    """Stationary AR(1) Gaussian construction; normal innovations only."""
    z = x.copy()
    for t in range(1, x.shape[-1]): z[..., t] = phi*z[..., t-1]+np.sqrt(1-phi**2)*x[..., t]
    return z


def _generate_original(rng, B, N, mean_delta=0., variance_delta=0., law='normal', phi=0., cluster_size=1, cluster_rho=0.):
    """Unchanged four-period scientific DGP and random-number draw order.

    Private compatibility entry point: inputs are expected to have been
    validated. Returns the two original ``(B, N, 4)`` panel arrays. Unit effects
    have variance .25 and shared time effects variance .09; treated error
    variance is ``1 + variance_delta*t`` and mean shift is ``mean_delta*t``.
    """
    times=np.arange(1,K+1)
    tau=rng.normal(0,.3,(B,1,K))
    panels=[]
    for g in (0,1):
        e=standard_errors(rng,(B,N,K),law)
        if phi:
            if law!='normal': raise ValueError('AR stress benchmark requires normal errors')
            e=ar_transform(e,phi)
        if cluster_size>1:
            if N%cluster_size or law!='normal': raise ValueError('balanced normal clusters required')
            c=ar_transform(rng.standard_normal((B,N//cluster_size,K)),phi)
            e=np.sqrt(cluster_rho)*np.repeat(c,cluster_size,axis=1)+np.sqrt(1-cluster_rho)*e
        if g: e=e*np.sqrt(1+variance_delta*times)[None,None,:]
        y=e+rng.normal(0,.5,(B,N,1))+tau
        if g: y=y+mean_delta*times[None,None,:]
        panels.append(y)
    return panels


def _validate_design(n_control: int, n_treated: int, n_periods: int, mean_delta: float,
                     variance_delta: float, law: str, phi: float, cluster_size: int,
                     cluster_rho: float) -> dict[str, Any]:
    n_control = _positive_integer(n_control, "n_control", 2)
    n_treated = _positive_integer(n_treated, "n_treated", 2)
    n_periods = _positive_integer(n_periods, "n_periods", 2)
    cluster_size = _positive_integer(cluster_size, "cluster_size")
    if not isinstance(law, str) or law not in _LAWS:
        raise ValueError(f"law must be one of {_LAWS}")
    converted = {}
    for name, value in (("mean_delta", mean_delta), ("variance_delta", variance_delta),
                        ("phi", phi), ("cluster_rho", cluster_rho)):
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite number")
        converted[name] = scalar
    if not -1 < converted["phi"] < 1:
        raise ValueError("phi must lie strictly between -1 and 1")
    if not 0 <= converted["cluster_rho"] <= 1:
        raise ValueError("cluster_rho must lie between 0 and 1")
    if (converted["phi"] != 0 or cluster_size > 1) and law != "normal":
        raise ValueError("Serial and clustered stress designs require normal innovations")
    if cluster_size == 1 and converted["cluster_rho"] != 0:
        raise ValueError("A nonzero cluster_rho requires cluster_size > 1")
    if n_control % cluster_size or n_treated % cluster_size:
        raise ValueError("cluster_size must divide both group sizes")
    with np.errstate(over="ignore", invalid="ignore"):
        times = np.arange(1, n_periods + 1)
        variance_path = 1 + converted["variance_delta"] * times
        mean_path = converted["mean_delta"] * times
    if not np.isfinite(variance_path).all() or np.any(variance_path <= 0):
        raise ValueError("Treated error variance 1 + variance_delta*time must be finite and positive in every period")
    if not np.isfinite(mean_path).all():
        raise ValueError("mean_delta*time must be finite in every period")
    return dict(n_control=n_control, n_treated=n_treated, n_periods=n_periods,
                law=law, cluster_size=cluster_size, **converted)


def _generate_panels(rng: np.random.Generator, count: int, design: dict) -> tuple[NDArray, NDArray]:
    if design["n_periods"] == 4 and design["n_control"] == design["n_treated"]:
        a, b = _generate_original(rng, count, design["n_control"], **{
            k: design[k] for k in ("mean_delta", "variance_delta", "law", "phi", "cluster_size", "cluster_rho")})
        return a, b
    k, m = design["n_periods"], design["cluster_size"]
    times = np.arange(1, k + 1)
    tau = rng.normal(0, .3, (count, 1, k))
    panels = []
    for group, n in enumerate((design["n_control"], design["n_treated"])):
        e = standard_errors(rng, (count, n, k), design["law"])
        if design["phi"]:
            e = ar_transform(e, design["phi"])
        if m > 1:
            c = ar_transform(rng.standard_normal((count, n // m, k)), design["phi"])
            e = np.sqrt(design["cluster_rho"]) * np.repeat(c, m, axis=1) + np.sqrt(1 - design["cluster_rho"]) * e
        if group:
            e = e * np.sqrt(1 + design["variance_delta"] * times)[None, None, :]
        y = e + rng.normal(0, .5, (count, n, 1)) + tau
        if group:
            y = y + design["mean_delta"] * times[None, None, :]
        panels.append(y)
    return panels[0], panels[1]


def simulate_panel(
    n_control: int = 200, n_treated: int = 200, n_periods: int = 4,
    mean_delta: float = 0., variance_delta: float = 0., law: str = "normal",
    phi: float = 0., cluster_size: int = 1, cluster_rho: float = 0.,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Generate one balanced, complete pretreatment panel in long form.

    Returns columns ``unit``, ``time`` (1 through K), ``treated`` (0 or 1),
    ``outcome`` and ``cluster``. Unit and cluster IDs are distinct across groups;
    rows are sorted by group, unit and time. ``cluster_size=1`` identifies units
    individually; larger values create contiguous, group-specific equal-size
    clusters. This generator does not create shared cross-group clusters.

    Unit effects are N(0,.25); a common N(0,.09) effect at each time is shared
    by both groups. Control error variance is one. Treated error variance is
    ``1 + variance_delta*time``, with mean departure ``mean_delta*time``.
    Innovations are standardized normal, uniform, t(5), Laplace or centered
    exponential. Serial AR(1) and common-cluster mixtures require normal errors;
    ``cluster_rho`` is the common-cluster variance share, applied before the
    treated variance multiplier. Unit, time and error innovations are mutually
    independent. The same law and dependence settings apply to both groups.

    Four-period equal-size inputs preserve the finalized study's generator.
    Unequal group sizes and other period counts are explicit extensions, not
    evidence that the published Monte Carlo results apply to those designs.
    """
    design = _validate_design(n_control, n_treated, n_periods, mean_delta,
                              variance_delta, law, phi, cluster_size, cluster_rho)
    y0, y1 = _generate_panels(_random_generator(random_state), 1, design)
    n0, n1, k, m = design["n_control"], design["n_treated"], design["n_periods"], design["cluster_size"]
    units = np.arange(n0 + n1)
    cluster_ids = np.concatenate((np.arange(n0) // m, n0 // m + np.arange(n1) // m))
    frame = pd.DataFrame({"unit": np.repeat(units, k), "time": np.tile(np.arange(1, k + 1), n0 + n1),
                          "treated": np.repeat(np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)], k),
                          "outcome": np.concatenate((y0[0].ravel(), y1[0].ravel())),
                          "cluster": np.repeat(cluster_ids, k)})
    frame.attrs["simulation"] = dict(design, random_state=None if random_state is None else int(random_state))
    return frame


def _wilson(rejections: int, repetitions: int) -> tuple[float, float]:
    z = 1.959963984540054
    p, a = rejections / repetitions, z*z / repetitions
    mid = (p + a / 2) / (1 + a)
    half = z * np.sqrt(p*(1-p)/repetitions + z*z/(4*repetitions**2)) / (1+a)
    return (0. if rejections == 0 else float(max(0., mid-half)),
            1. if rejections == repetitions else float(min(1., mid+half)))


def monte_carlo(
    *, repetitions: int = 100, n_control: int = 200, n_treated: int = 200,
    n_periods: int = 4, mean_delta: float = 0., variance_delta: float = 0.,
    law: str = "normal", phi: float = 0., cluster_size: int = 1,
    cluster_rho: float = 0., random_state: int | None = 0, batch_size: int = 100,
    reference: int = 0, alpha: float = .05, fisher_valid: bool = False,
) -> dict[str, Any]:
    """Run a configurable analytic Monte Carlo with bounded panel batching.

    Returns ``summary`` (one DataFrame row for mean, variance and Fisher),
    ``raw`` (all observed statistics, p-values, validity and rejection flags),
    and ``configuration``. Rates, plug-in Monte Carlo SEs and pointwise 95%
    Wilson intervals are fractions, not percentages. Rejection uses ``p<=alpha``.
    Every scheduled repetition remains in the denominator; failures are reported
    separately and are never rerun. An all-scheduled rejection rate with failures
    is a recorded rejection frequency, not a claim of calibrated power/size.
    Disabled Fisher is distinguished from numerical failure.

    Fisher is off by default. Opt-in is permitted only for the symmetric laws
    in this generator, using the specified iid-unit/cluster symmetry argument;
    no exact finite-sample calibration is claimed. Exponential Fisher is not
    enabled. Cluster sizes >1 select group-specific cluster covariance. A
    two-period variance test is degenerate and its failures remain explicit.

    Only current data batches are held in memory; raw results scale with the
    requested repetition count. Seed, batch size and all design settings must
    remain fixed for replay. This helper does not run any published design grid,
    bootstrap calibration study, or historical empirical application.
    """
    repetitions = _positive_integer(repetitions, "repetitions")
    batch_size = _positive_integer(batch_size, "batch_size")
    design = _validate_design(n_control, n_treated, n_periods, mean_delta,
                              variance_delta, law, phi, cluster_size, cluster_rho)
    reference = _positive_integer(reference, "reference", 0)
    if reference >= n_periods:
        raise ValueError("reference must index an observed period")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a finite number strictly between zero and one") from exc
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha_value) or not 0 < alpha_value < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")
    alpha = alpha_value
    if not isinstance(fisher_valid, (bool, np.bool_)):
        raise ValueError("fisher_valid must be Boolean")
    if fisher_valid and law == "exponential":
        raise ValueError("Fisher calibration is not justified for the asymmetric exponential design")
    m = design["cluster_size"]
    if m > 1 and min(n_control // m, n_treated // m) < 2:
        raise ValueError("Cluster inference requires at least two clusters in each group")
    rng = _random_generator(random_state)
    pieces: dict[str, list[NDArray]] = {}
    for start in range(0, repetitions, batch_size):
        y0, y1 = _generate_panels(rng, min(batch_size, repetitions-start), design)
        fit = _core.analyze_batch(y0, y1, reference=reference, fisher_valid=fisher_valid,
                                  cluster_size=m if m > 1 else None)
        for name in ("W_mean", "W_var", "W_sum", "S_fisher", "p_mean", "p_var", "p_fisher",
                     "valid_mean", "valid_var", "valid_fisher"):
            pieces.setdefault(name, []).append(fit[name])
    raw = {name: np.concatenate(parts) for name, parts in pieces.items()}
    rows = []
    for name, suffix in (("mean", "mean"), ("variance", "var"), ("fisher", "fisher")):
        enabled = name != "fisher" or bool(fisher_valid)
        valid = raw[f"valid_{suffix}"] & np.isfinite(raw[f"p_{suffix}"])
        rejection = valid & (raw[f"p_{suffix}"] <= alpha)
        raw[f"reject_{suffix}"] = rejection
        count, nvalid = int(rejection.sum()), int(valid.sum())
        rate = count / repetitions
        lo, hi = _wilson(count, repetitions)
        rows.append(dict(test=name, enabled=enabled, scheduled_repetitions=repetitions,
                         valid_repetitions=nvalid, failed_repetitions=repetitions-nvalid if enabled else 0,
                         disabled_repetitions=0 if enabled else repetitions,
                         rejections=count if enabled else 0, rejection_rate=rate if enabled else np.nan,
                         mcse=np.sqrt(rate*(1-rate)/repetitions) if enabled else np.nan,
                         wilson95_low=lo if enabled else np.nan, wilson95_high=hi if enabled else np.nan))
    return {"summary": pd.DataFrame(rows), "raw": raw,
            "configuration": dict(design, repetitions=repetitions, batch_size=batch_size,
                                  random_state=None if random_state is None else int(random_state),
                                  reference=reference, alpha=float(alpha), fisher_valid=bool(fisher_valid))}
