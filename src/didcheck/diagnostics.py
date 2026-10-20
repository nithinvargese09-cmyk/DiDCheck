"""Public interface to the paper's fixed-window variance-ratio diagnostic."""
from __future__ import annotations

from typing import Any, Sequence
from numbers import Real
import numpy as np
import pandas as pd

from . import _core
from .results import DiagnosticResult, TestResult
from .utils import cluster_labels, integer, prepare_panel, validate_arrays


def _test(stat: Any, p: Any, df: int | None, valid: Any, alpha: float,
          calibration: str = "asymptotic chi-square") -> TestResult:
    good = bool(bool(valid) and not np.isnan(stat) and np.isfinite(p))
    return TestResult(float(stat), float(p), df, bool(p <= alpha) if good else None, good, calibration)


def diagnose_arrays(control: Any, treated: Any, *, reference: int = 0, alpha: float = .05,
                    fisher: bool = False, fisher_justification: str | None = None,
                    bootstrap: int = 0, random_state: int | None = 0, batch_size: int = 100,
                    cluster_size: int | tuple[int, int] | None = None,
                    clusters: tuple[Any, Any] | None = None) -> DiagnosticResult:
    """Analyze complete control and treated trajectories of shape (N_g, K).

    K >= 3; observations must be from a common pretreatment window. The default
    sampling assumption is iid unit trajectories in independent groups.
    ``cluster_size`` selects equal-size, contiguous, separately sampled group
    clusters. ``clusters`` instead selects whole iid clusters with labels shared
    across groups when appropriate. These sampling designs are mutually exclusive.

    ``bootstrap`` is the scheduled resample count (zero disables it). Any invalid
    required resample makes that component's bootstrap calibration unavailable.
    Fisher requires an explicit joint-independence justification; its presence
    records a user assumption, not a statistical check of independence.
    """
    y0, y1 = validate_arrays(control, treated)
    reference = integer(reference, "reference")
    if reference >= y0.shape[1]:
        raise ValueError("reference is outside the common period window")
    bootstrap = integer(bootstrap, "bootstrap")
    batch_size = integer(batch_size, "batch_size", 1)
    if random_state is not None:
        random_state = integer(random_state, "random_state")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if not isinstance(fisher, (bool, np.bool_)):
        raise ValueError("fisher must be a boolean")
    if fisher and (not isinstance(fisher_justification, str) or not fisher_justification.strip()):
        raise ValueError("Fisher requires fisher_justification for joint asymptotic independence")
    if clusters is not None and cluster_size is not None:
        raise ValueError("Choose clusters or cluster_size, not both")
    notes = ["Finite-sample calibration and power may be limited; numerical validity does not establish inferential assumptions."]
    sampling = "independent units within independent groups"
    if clusters is not None:
        if not isinstance(clusters, (tuple, list)) or len(clusters) != 2:
            raise ValueError("clusters must contain control and treated label arrays")
        clusters = (cluster_labels(clusters[0], len(y0)), cluster_labels(clusters[1], len(y1)))
        fit = _core.analyze_clustered(y0, y1, *clusters, reference=reference, fisher_valid=fisher)
        sampling = "whole iid clusters, possibly shared across groups"
        notes.append("Shared/unequal-cluster inference requires additional cluster regularity; the paper's Monte Carlo study does not validate all such designs.")
    else:
        if cluster_size is not None:
            sizes = (cluster_size, cluster_size) if np.isscalar(cluster_size) else tuple(cluster_size)
            if len(sizes) != 2:
                raise ValueError("cluster_size must be an integer or a pair")
            cluster_size = tuple(integer(x, "cluster_size", 1) for x in sizes)
            sampling = "equal-size whole clusters sampled separately within groups"
            notes.append("Cluster calibration requires many independent clusters and no dominant cluster; finite-cluster corrections are not exact laws.")
        fit = _core.analyze(y0, y1, reference=reference, fisher_valid=fisher, cluster_size=cluster_size)
    if not fit["valid_var"]:
        notes.append("Variance inference is unavailable: positive variances and a finite nonsingular contrast covariance are required.")
    if not fit["valid_mean"]:
        notes.append("Mean inference is unavailable: a finite nonsingular contrast covariance is required.")
    if fisher:
        notes.append("Fisher calibration relies on the supplied joint-independence justification: " + fisher_justification.strip())
    else:
        notes.append("Fisher's combination is disabled; component p-values need not be independent.")
    boot = None
    combined = _test(fit["W_sum"], np.nan, None, False, alpha, "uncalibrated sum; request centered bootstrap")
    if bootstrap:
        from .bootstrap import centered_bootstrap
        boot = centered_bootstrap(y0, y1, repetitions=bootstrap, random_state=random_state,
                                  reference=reference, batch_size=batch_size, cluster_size=cluster_size, clusters=clusters)
        b = boot["sum"]
        combined = _test(b["statistic"], b["pvalue"], None, b["valid"], alpha, "centered pairs bootstrap")
        for key in ["mean", "variance", "sum"]:
            if boot[key]["failed_draws"]:
                notes.append(f"{key} bootstrap unavailable: {boot[key]['failed_draws']} of {bootstrap} required draws failed; none were replaced or discarded.")
    k = y0.shape[1]
    return DiagnosticResult(
        mean=_test(fit["W_mean"], fit["p_mean"], k-1, fit["valid_mean"], alpha),
        variance=_test(fit["W_var"], fit["p_var"], k-1, fit["valid_var"], alpha),
        fisher=_test(fit["S_fisher"], fit["p_fisher"], 4, fit["valid_fisher"], alpha),
        combined=combined, variance_ratios=fit["ratios"], log_variance_ratios=fit["log_ratios"],
        mean_contrast=fit["h_mean"], variance_contrast=fit["h_var"], mean_covariance=fit["cov_mean"],
        variance_covariance=fit["cov_var"], warnings=tuple(notes), alpha=float(alpha),
        n_control=len(y0), n_treated=len(y1), periods=tuple(range(k)), reference=reference,
        bootstrap=boot, metadata={"sampling": sampling, "fisher_justification": fisher_justification})


def variance_test(data: pd.DataFrame, *, outcome: str, unit: str, time: str, treatment: str,
                  pre_periods: Sequence[Any], reference: Any = None, missing: str = "raise",
                  cluster: str | None = None, cluster_design: str | None = None,
                  alpha: float = .05, fisher: bool = False, fisher_justification: str | None = None,
                  bootstrap: int = 0, random_state: int | None = 0, batch_size: int = 100) -> DiagnosticResult:
    """Apply the paper's diagnostic to an explicitly selected pretreatment panel.

    ``treatment`` is a time-invariant 0/1 group indicator. ``pre_periods`` is the
    ordered common window, with at least three periods. Dates are labels, not
    inferred treatment times. Default reference is its first element. Missing
    trajectories raise unless ``missing='drop_units'`` is explicitly selected.
    Cluster designs: ``shared`` or ``stratified_equal_size``; see the array API.
    """
    panel = prepare_panel(data, outcome=outcome, unit=unit, time=time, treatment=treatment,
                          pre_periods=pre_periods, missing=missing, cluster=cluster, cluster_design=cluster_design)
    ref = panel.periods[0] if reference is None else reference
    if ref not in panel.periods:
        raise ValueError("reference must name a selected pretreatment period")
    result = diagnose_arrays(panel.control, panel.treated, reference=panel.periods.index(ref), alpha=alpha,
                             fisher=fisher, fisher_justification=fisher_justification, bootstrap=bootstrap,
                             random_state=random_state, batch_size=batch_size,
                             cluster_size=panel.cluster_size, clusters=panel.clusters)
    result.periods, result.reference = panel.periods, ref
    result.metadata.update(control_units=panel.control_units, treated_units=panel.treated_units,
                           dropped_units=panel.dropped_units, missing_policy=missing)
    if panel.dropped_units:
        result.warnings += (f"Complete-case analysis explicitly removed {len(panel.dropped_units)} units; selective attrition may change the target.",)
    return result
