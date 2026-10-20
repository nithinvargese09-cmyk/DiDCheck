"""Synthetic illustration only: 40 separate clusters of five units per group.

Run after installing the local package:
    python examples/clustered_panel.py
    python examples/clustered_panel.py --plot

The simulated clusters are independent across blocks and group-specific. This
example does not establish calibration for an empirical or shared-cluster design.
"""
from __future__ import annotations

import argparse

from didcheck import simulate_panel, variance_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=199, help="Scheduled whole-cluster draws")
    parser.add_argument("--seed", type=int, default=87, help="Fixed seed for this synthetic example")
    parser.add_argument("--plot", action="store_true", help="Show descriptive plots (requires Matplotlib)")
    args = parser.parse_args()

    panel = simulate_panel(
        n_control=200,
        n_treated=200,
        n_periods=4,
        mean_delta=0.0,
        variance_delta=0.20,
        law="normal",
        phi=0.5,
        cluster_size=5,
        cluster_rho=0.4,
        random_state=args.seed,
    )
    result = variance_test(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treatment="treated",
        pre_periods=[1, 2, 3, 4],
        cluster="cluster",
        cluster_design="stratified_equal_size",
        bootstrap=args.bootstrap,
        random_state=args.seed + 1,
        fisher=False,
    )
    counts = panel.groupby("treated")["cluster"].nunique()
    print("Synthetic group-specific cluster example; no empirical application.")
    print("Independent clusters per group:", counts.to_dict())
    print(result.summary())
    print("Within-cluster dependence is retained by resampling complete blocks.")
    print("This single panel does not establish finite-sample calibration or power.")

    if args.plot:
        import matplotlib.pyplot as plt
        from didcheck.plotting import plot_bootstrap, plot_variance_ratios

        plot_variance_ratios(result)
        if result.bootstrap is not None:
            plot_bootstrap(result, component="sum")
        plt.show()


if __name__ == "__main__":
    main()
