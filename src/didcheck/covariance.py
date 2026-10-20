"""Covariance utilities for full temporal dependence, not diagonal shortcuts."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.stats import chi2
from . import _core


def within_unit_demean(outcomes: Any) -> np.ndarray:
    """Subtract each unit's common-window time mean from a finite 2-D array."""
    if np.iscomplexobj(outcomes):
        raise ValueError("Outcomes must be real, not complex")
    y = np.asarray(outcomes, dtype=float)
    if y.ndim != 2 or y.shape[1] < 2 or not np.isfinite(y).all():
        raise ValueError("Expected a finite (units, periods) array with >=2 periods")
    return y - y.mean(axis=1, keepdims=True)


def contrast_matrix(n_periods: int, reference: int = 0) -> np.ndarray:
    """Period-minus-reference contrasts, in the order of nonreference periods."""
    from .utils import integer
    return _core.contrast_matrix(integer(n_periods, "n_periods", 2), integer(reference, "reference"))


def wald_statistic(contrast: Any, covariance: Any) -> tuple[float, float, bool]:
    """Compute a full-covariance Wald statistic and asymptotic chi-square tail.

    Invalid covariance returns (NaN, NaN, False). No pseudoinverse or estimated
    rank is used; degrees of freedom are the supplied contrast dimension.
    """
    if np.iscomplexobj(contrast) or np.iscomplexobj(covariance):
        raise ValueError("Contrast and covariance must be real, not complex")
    h, v = np.asarray(contrast, float), np.asarray(covariance, float)
    if h.ndim != 1 or h.size == 0 or v.shape != (h.size, h.size):
        raise ValueError("Expected contrast (p,) and covariance (p,p), p>=1")
    w, good = _core.wald_batch(h[None], v[None])
    return float(w[0]), float(chi2.sf(w[0], len(h))), bool(good[0])
