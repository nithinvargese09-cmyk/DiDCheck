"""Null-centered pairs bootstrap for the mean, variance and sum diagnostics.

The sum is the sum of two component Wald statistics, not a full joint Wald
statistic. Every draw is retained. A failed required draw makes the corresponding
bootstrap p-value unavailable; draws are never deleted, replaced or retried.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _random_generator(random_state: int | None) -> np.random.Generator:
    if random_state is not None:
        random_state = _positive_integer(random_state, "random_state", minimum=0)
    return np.random.default_rng(random_state)


def _arrays(y0: ArrayLike, y1: ArrayLike, reference: int) -> tuple[NDArray, NDArray]:
    if np.iscomplexobj(y0) or np.iscomplexobj(y1):
        raise ValueError("Outcomes must be real-valued")
    a, b = np.asarray(y0, dtype=float), np.asarray(y1, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("y0 and y1 must be complete (units, periods) arrays with the same periods")
    if min(len(a), len(b)) < 2 or a.shape[1] < 2:
        raise ValueError("At least two units per group and two periods are required")
    reference = _positive_integer(reference, "reference", minimum=0)
    if reference >= a.shape[1]:
        raise ValueError("reference must index an observed period")
    return a, b


def _cluster_sizes(value: int | tuple[int, int] | None, n0: int, n1: int) -> tuple[int, int] | None:
    if value is None:
        return None
    sizes = value if isinstance(value, tuple) else (value, value)
    if len(sizes) != 2:
        raise ValueError("cluster_size must be an integer or a two-integer tuple")
    checked = tuple(_positive_integer(v, "cluster_size") for v in sizes)
    for n, m in zip((n0, n1), checked):
        if n % m or n // m < 2:
            raise ValueError("cluster_size must divide group size and leave at least two clusters per group")
    return checked


def _labels(values: Sequence[Any], n: int) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Cluster labels must be one-dimensional sequences, not scalar strings")
    try:
        labels = list(values)
    except TypeError as exc:
        raise ValueError("Cluster labels must be one-dimensional sequences") from exc
    if len(labels) != n:
        raise ValueError("Each unit must have exactly one cluster label")
    for label in labels:
        try:
            hash(label)
            missing = label is None or bool(label != label)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cluster labels must be nonmissing, hashable scalar identifiers") from exc
        if missing:
            raise ValueError("Cluster labels must not be missing")
    return labels


def _resample_shared_clusters(
    y0: NDArray, y1: NDArray, labels0: Sequence[Any], labels1: Sequence[Any],
    rng: np.random.Generator, selected: Sequence[Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Draw pooled iid clusters, giving repeated copies distinct new identities.

    ``selected`` is a deterministic checking hook containing exactly G known
    labels. Group denominators may vary, and a draw can lose a group entirely.
    Such draws are retained as failures by :func:`centered_bootstrap`.
    """
    c0, c1, labels = _core._cluster_labels(labels0, labels1, len(y0), len(y1))
    if selected is None:
        selected = [labels[j] for j in rng.integers(len(labels), size=len(labels))]
    if len(selected) != len(labels) or any(label not in labels for label in selected):
        raise ValueError("selected must contain exactly G known cluster labels")
    blocks0, blocks1, ids0, ids1 = [], [], [], []
    for copy_id, label in enumerate(selected):
        i0 = [j for j, value in enumerate(c0) if value == label]
        i1 = [j for j, value in enumerate(c1) if value == label]
        blocks0.append(y0[i0])
        blocks1.append(y1[i1])
        ids0.extend([copy_id] * len(i0))
        ids1.extend([copy_id] * len(i1))
    return np.concatenate(blocks0), np.concatenate(blocks1), np.asarray(ids0), np.asarray(ids1)


def _centered_components(fit: dict, observed: dict) -> tuple[dict, dict]:
    wm, okm = _core.wald_batch(fit["h_mean"] - observed["h_mean"], fit["cov_mean"])
    wv, okv = _core.wald_batch(fit["h_var"] - observed["h_var"], fit["cov_var"])
    okm &= fit["valid_mean"]
    okv &= fit["valid_var"]
    total, okt = _core.centered_sum_batch(fit, observed)
    wm[~okm], wv[~okv] = np.nan, np.nan
    return {"mean": wm, "variance": wv, "sum": total}, {"mean": okm, "variance": okv, "sum": okt}


def _tail(statistic: float, statistics: NDArray, valid_draws: NDArray) -> dict[str, Any]:
    flags = np.asarray(valid_draws, dtype=bool) & np.isfinite(statistics)
    failed = int(np.count_nonzero(~flags))
    exceedances = int(np.count_nonzero(statistics[flags] >= statistic)) if np.isfinite(statistic) else None
    valid = bool(np.isfinite(statistic) and failed == 0)
    return {
        "statistic": float(statistic),
        "pvalue": float((1 + exceedances) / (len(statistics) + 1)) if valid else float("nan"),
        "statistics": statistics,
        "valid_draws": flags,
        "failed_draws": failed,
        "exceedances": exceedances,
        "valid": valid,
    }


def centered_bootstrap(
    y0: ArrayLike, y1: ArrayLike, *, repetitions: int = 999,
    random_state: int | None = 0, reference: int = 0, batch_size: int = 100,
    cluster_size: int | tuple[int, int] | None = None,
    clusters: tuple[Sequence[Any], Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate mean, variance and their sum with centered pairs resampling.

    Parameters
    ----------
    y0, y1
        Complete control/treated trajectory matrices, shape ``(N_g, K)``. Units
        must share the same period order. Nonfinite or degenerate numerical
        inputs produce unavailable results, not replacement observations.
    repetitions, random_state, reference, batch_size
        Positive draw/batch counts, a nonnegative seed (or ``None``), and a
        zero-based reference period. The seed *and* batch size determine the
        within-group RNG sequence. Keep both fixed for exact replay. Pooled
        shared-cluster draws are processed one at a time.
    cluster_size
        Integer or ``(control_size, treated_size)`` for equal-size contiguous
        group-specific clusters, resampled separately within each group. Uses
        the documented ``G_g/(G_g-1)`` covariance correction. ``None`` resamples
        whole units without a cluster correction; explicit ``1`` uses the
        finite-cluster convention and is not identical to ``None``.
    clusters
        ``(control_labels, treated_labels)`` for pooled iid clusters, possibly
        shared across groups. Equal labels designate the same cluster. Both
        groups are copied together; each selected cluster copy gets a fresh
        identifier. Uses pooled ``G/(G-1)`` covariance correction. Do not use
        pooled resampling for a stratified sampling design. Mutually exclusive
        with ``cluster_size``.

    Returns
    -------
    dict
        ``mean``, ``variance`` and ``sum`` each hold the observed ``statistic``,
        ``pvalue``, all centered ``statistics``, Boolean ``valid_draws`` mask,
        integer ``failed_draws``, ``exceedances`` and overall ``valid`` flag.
        An invalid observed statistic or any required failed draw gives NaN
        p-value. Exceedances, when the observation is finite, count valid draws
        only for diagnosis; they are never used to drop failures from a p-value.
        Top-level fields record ``repetitions``, ``random_state``, ``resampling``
        and ``batch_size``. P-values use ``(1 + count(star >= observed))/(B+1)``;
        rejection is inclusive at the chosen level.

    Notes
    -----
    Each draw uses the same trajectories for both components and centers each
    restriction at its original estimate, using the resampled covariance. This
    preserves dependence in the sum without Fisher independence assumptions.
    Validity requires appropriate iid-unit/cluster resampling, a joint bootstrap
    limit, consistent covariances and a continuous limiting null law. Numerical
    flags and plus-one counting do not verify these conditions or imply exact
    finite-sample size. Two-period variance degeneracy is retained as failure.
    """
    repetitions = _positive_integer(repetitions, "repetitions")
    batch_size = _positive_integer(batch_size, "batch_size")
    rng = _random_generator(random_state)
    y0, y1 = _arrays(y0, y1, reference)
    if cluster_size is not None and clusters is not None:
        raise ValueError("cluster_size and clusters are mutually exclusive")
    sizes = _cluster_sizes(cluster_size, len(y0), len(y1))
    values = {name: np.full(repetitions, np.nan) for name in ("mean", "variance", "sum")}
    flags = {name: np.zeros(repetitions, dtype=bool) for name in values}
    if clusters is not None:
        if not isinstance(clusters, (tuple, list)) or len(clusters) != 2:
            raise ValueError("clusters must contain control and treated label sequences")
        c0, c1 = _labels(clusters[0], len(y0)), _labels(clusters[1], len(y1))
        observed = _core.analyze_clustered(y0, y1, c0, c1, reference)
        resampling = "pooled iid whole clusters; shared groups copied together"
        for j in range(repetitions):
            a, b, ca, cb = _resample_shared_clusters(y0, y1, c0, c1, rng)
            # A pooled draw can leave fewer than two units in a group. It is
            # retained in the scheduled denominator, never redrawn.
            if min(len(a), len(b)) < 2:
                continue
            fit = _core.analyze_clustered(a, b, ca, cb, reference)
            batch = {key: np.asarray(fit[key])[None] for key in (
                "h_mean", "h_var", "cov_mean", "cov_var", "valid_mean", "valid_var")}
            v, ok = _centered_components(batch, observed)
            for name in values:
                values[name][j], flags[name][j] = v[name][0], ok[name][0]
    else:
        observed = _core.analyze(y0, y1, reference, cluster_size=sizes)
        resampling = "whole units within each group" if sizes is None else "whole equal-size contiguous clusters within each group"
        m0, m1 = (1, 1) if sizes is None else sizes
        g0, g1 = len(y0) // m0, len(y1) // m1
        for start in range(0, repetitions, batch_size):
            count = min(batch_size, repetitions - start)
            i0 = rng.integers(g0, size=(count, g0))
            i1 = rng.integers(g1, size=(count, g1))
            if m0 > 1:
                i0 = (i0[..., None] * m0 + np.arange(m0)).reshape(count, len(y0))
            if m1 > 1:
                i1 = (i1[..., None] * m1 + np.arange(m1)).reshape(count, len(y1))
            fit = _core.analyze_batch(y0[i0], y1[i1], reference, cluster_size=sizes)
            v, ok = _centered_components(fit, observed)
            for name in values:
                values[name][start:start + count] = v[name]
                flags[name][start:start + count] = ok[name]
    result = {name: _tail(observed[key], values[name], flags[name]) for name, key in (
        ("mean", "W_mean"), ("variance", "W_var"), ("sum", "W_sum"))}
    result.update(repetitions=repetitions, random_state=None if random_state is None else int(random_state),
                  resampling=resampling, batch_size=batch_size)
    return result
