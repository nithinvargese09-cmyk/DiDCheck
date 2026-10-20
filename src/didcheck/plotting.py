"""Descriptive figures without unsupported simultaneous confidence bands."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
if TYPE_CHECKING:
    from .results import DiagnosticResult


def plot_variance_ratios(result: DiagnosticResult, ax: Any = None) -> Any:
    """Plot sample variance ratios; points are descriptive, without intervals."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    x = np.arange(len(result.periods))
    ax.plot(x, result.variance_ratios, marker="o")
    ax.set_xticks(x, [str(p) for p in result.periods])
    ax.set(xlabel="Pretreatment period", ylabel="Treated/control demeaned variance ratio",
           title="Descriptive variance-ratio path")
    return ax


def plot_bootstrap(result: DiagnosticResult, component: str = "sum", ax: Any = None) -> Any:
    """Plot finite centered bootstrap statistics, disclosing failed draw counts."""
    import matplotlib.pyplot as plt
    if result.bootstrap is None:
        raise ValueError("No bootstrap draws available; request bootstrap in the diagnostic")
    if component not in {"mean", "variance", "sum"}:
        raise ValueError("component must be mean, variance, or sum")
    b = result.bootstrap[component]
    values = np.asarray(b["statistics"])
    if ax is None:
        _, ax = plt.subplots()
    finite = values[np.isfinite(values)]
    if finite.size:
        ax.hist(finite, bins="auto", alpha=.7)
    if np.isfinite(b["statistic"]):
        ax.axvline(b["statistic"], color="black", linestyle="--", label="Observed")
        ax.legend()
    ax.set(xlabel="Centered bootstrap statistic", ylabel="Draw count",
           title=f"{component}: {len(values)} scheduled; {b['failed_draws']} invalid")
    return ax
