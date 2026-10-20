"""Explicit validation and conversion of a common-window, two-group panel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class PreparedPanel:
    """Validated complete trajectories and their sampling metadata."""
    control: np.ndarray
    treated: np.ndarray
    periods: tuple[Any, ...]
    control_units: tuple[Any, ...]
    treated_units: tuple[Any, ...]
    dropped_units: tuple[Any, ...]
    clusters: tuple[np.ndarray, np.ndarray] | None = None
    cluster_size: tuple[int, int] | None = None


def integer(value: Any, name: str, minimum: int = 0) -> int:
    """Accept an integer parameter without silently truncating floats or bools."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_arrays(control: Any, treated: Any) -> tuple[np.ndarray, np.ndarray]:
    """Require finite (units, periods) arrays on the same window, K >= 3."""
    if np.iscomplexobj(control) or np.iscomplexobj(treated):
        raise ValueError("Outcomes must be real, not complex")
    y0, y1 = np.asarray(control, dtype=float), np.asarray(treated, dtype=float)
    if y0.ndim != 2 or y1.ndim != 2 or y0.shape[1] != y1.shape[1]:
        raise ValueError("control and treated must be (units, periods) arrays with the same periods")
    if min(len(y0), len(y1)) < 2:
        raise ValueError("At least two units per group are required")
    if y0.shape[1] < 3:
        raise ValueError("The variance diagnostic requires at least three pretreatment periods")
    if not np.isfinite(y0).all() or not np.isfinite(y1).all():
        raise ValueError("Outcomes must be finite and trajectories complete")
    return y0, y1


def cluster_labels(values: Any, n: int) -> np.ndarray:
    """Validate one nonmissing, hashable cluster identifier per trajectory."""
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("Each unit requires one scalar cluster label")
    for label in labels:
        try:
            hash(label)
            missing = bool(pd.isna(label))
        except (TypeError, ValueError):
            raise ValueError("Cluster identifiers must be nonmissing hashable scalars") from None
        if missing:
            raise ValueError("Cluster identifiers must not be missing")
    return labels


def prepare_panel(data: pd.DataFrame, *, outcome: str, unit: str, time: str,
                  treatment: str, pre_periods: Sequence[Any], missing: str = "raise",
                  cluster: str | None = None, cluster_design: str | None = None) -> PreparedPanel:
    """Convert an explicitly selected balanced pretreatment window to arrays.

    Treatment denotes time-invariant group membership, not treatment receipt.
    No imputation or automatic period selection is performed. ``drop_units``
    removes entire incomplete trajectories and reports their identifiers.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    columns = [outcome, unit, time, treatment] + ([cluster] if cluster else [])
    if len(set(columns)) != len(columns):
        raise ValueError("Outcome, unit, time, treatment and cluster columns must be distinct")
    if not data.columns.is_unique or any(c not in data.columns for c in columns):
        raise ValueError("Input columns must be unique and all requested columns must exist")
    if isinstance(pre_periods, (str, bytes)) or pre_periods is None:
        raise ValueError("Supply an ordered list of at least three explicit pretreatment periods")
    try:
        periods = tuple(pre_periods)
    except TypeError:
        raise ValueError("Supply an ordered list of explicit pretreatment periods") from None
    try:
        good_periods = len(periods) >= 3 and len(set(periods)) == len(periods) and not any(bool(pd.isna(t)) for t in periods)
    except (TypeError, ValueError):
        good_periods = False
    if not good_periods:
        raise ValueError("pre_periods must contain at least three unique nonmissing scalar periods")
    if missing not in {"raise", "drop_units"}:
        raise ValueError("missing must be 'raise' or 'drop_units'")
    if cluster is None and cluster_design is not None:
        raise ValueError("cluster_design requires a cluster column")
    if cluster is not None and cluster_design not in {"shared", "stratified_equal_size"}:
        raise ValueError("Choose cluster_design='shared' or 'stratified_equal_size' explicitly")
    d = data.loc[:, columns].copy()
    if d.empty or d[[unit, time, treatment]].isna().any().any():
        raise ValueError("Unit, time and treatment identifiers must be nonmissing")
    if not d[treatment].isin([0, 1]).all():
        raise ValueError("treatment must be binary 0/1 group membership")
    if (d.groupby(unit, sort=False)[treatment].nunique() != 1).any():
        raise ValueError("Treatment group membership must be time-invariant; staggered treatment is unsupported")
    if cluster:
        cluster_labels(d[cluster].to_numpy(), len(d))
        if (d.groupby(unit, sort=False)[cluster].nunique() != 1).any():
            raise ValueError("A unit's cluster must be constant across periods")
    selected = d[d[time].isin(periods)].copy()
    if selected.duplicated([unit, time]).any():
        raise ValueError("Duplicate unit-time observations in the pretreatment window")
    selected[outcome] = pd.to_numeric(selected[outcome], errors="raise")
    if np.iscomplexobj(selected[outcome].to_numpy()):
        raise ValueError("Outcomes must be real, not complex")
    if np.isinf(selected[outcome].to_numpy(dtype=float)).any():
        raise ValueError("Infinite outcomes are not allowed")
    units = sorted(d[unit].unique(), key=lambda x: (type(x).__name__, repr(x)))
    wide = selected.pivot(index=unit, columns=time, values=outcome).reindex(index=units, columns=list(periods))
    bad = wide.isna().any(axis=1)
    dropped = tuple(wide.index[bad])
    if dropped and missing == "raise":
        raise ValueError(f"{len(dropped)} incomplete unit trajectories; use missing='drop_units' only for an explicit complete-case analysis")
    wide = wide.loc[~bad]
    info = d.drop_duplicates(unit).set_index(unit).reindex(wide.index)
    ids = [list(info.index[info[treatment] == g]) for g in [0, 1]]
    sizes = None
    labels = None
    if cluster:
        raw_labels = [info.loc[ix, cluster].to_numpy(dtype=object) for ix in ids]
        if cluster_design == "stratified_equal_size":
            if set(raw_labels[0]) & set(raw_labels[1]):
                raise ValueError("Stratified clusters cannot share identifiers across groups; use shared for genuinely shared clusters")
            sizes_list = []
            for g in [0, 1]:
                blocks = list(dict.fromkeys(raw_labels[g]))
                counts = [sum(x == c for x in raw_labels[g]) for c in blocks]
                if len(blocks) < 2 or len(set(counts)) != 1:
                    raise ValueError("Stratified designs require >=2 equal-size clusters within each group")
                sizes_list.append(counts[0])
                ids[g] = [u for c in blocks for u in ids[g] if info.loc[u, cluster] == c]
            sizes = tuple(sizes_list)
        else:
            labels = (raw_labels[0], raw_labels[1])
    y0, y1 = validate_arrays(wide.loc[ids[0]].to_numpy(dtype=float), wide.loc[ids[1]].to_numpy(dtype=float))
    return PreparedPanel(y0, y1, periods, tuple(ids[0]), tuple(ids[1]), dropped, labels, sizes)
