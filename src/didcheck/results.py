"""Structured results; rejection is evidence against a specified moment null."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TestResult:
    """A statistic and its calibration; ``valid`` concerns computation only."""
    statistic: float
    pvalue: float
    df: int | None
    reject: bool | None
    valid: bool
    calibration: str = "asymptotic chi-square"


@dataclass
class DiagnosticResult:
    """Mean/variance diagnostics and transparent sampling/validity metadata."""
    mean: TestResult
    variance: TestResult
    fisher: TestResult
    combined: TestResult
    variance_ratios: np.ndarray
    log_variance_ratios: np.ndarray
    mean_contrast: np.ndarray
    variance_contrast: np.ndarray
    mean_covariance: np.ndarray
    variance_covariance: np.ndarray
    warnings: tuple[str, ...]
    alpha: float
    n_control: int
    n_treated: int
    periods: tuple[Any, ...]
    reference: Any
    bootstrap: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return analytic tests and, when requested, centered bootstrap tests."""
        rows = []
        for name in ["mean", "variance", "fisher", "combined"]:
            t = getattr(self, name)
            rows.append(dict(test=name, statistic=t.statistic, pvalue=t.pvalue, df=t.df,
                             reject=t.reject, valid=t.valid, calibration=t.calibration))
        if self.bootstrap is not None:
            for name in ["mean", "variance"]:
                b = self.bootstrap[name]
                rows.append(dict(test="bootstrap_" + name, statistic=b["statistic"],
                                 pvalue=b["pvalue"], df=None, reject=bool(b["pvalue"] <= self.alpha) if b["valid"] else None,
                                 valid=b["valid"], calibration="centered pairs bootstrap"))
        return pd.DataFrame(rows).set_index("test")

    def summary(self) -> str:
        """Human-readable report without treating nonrejection as validation."""
        lines = ["DiDCheck: pretreatment moment diagnostics",
                 f"Units: control={self.n_control}, treated={self.n_treated}; periods={len(self.periods)}; alpha={self.alpha:g}",
                 self.to_frame().to_string(),
                 "Variance null: constant treated/control variance ratio after within-unit demeaning.",
                 "Nonrejection does not establish parallel trends, equivalence, or causal identification."]
        if self.bootstrap is not None:
            lines.append(f"Bootstrap: {self.bootstrap['repetitions']} scheduled draws; {self.bootstrap['resampling']}.")
        lines.extend("Note: " + w for w in self.warnings)
        return "\n".join(lines)
