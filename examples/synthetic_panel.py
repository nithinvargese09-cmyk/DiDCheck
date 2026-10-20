"""Synthetic illustration only: independent units, four pretreatment periods.

Run after installing the local package:
    python examples/synthetic_panel.py
    python examples/synthetic_panel.py --plot

A single generated panel illustrates the API; it is not empirical evidence or
a Monte Carlo calibration study. No desired rejection outcome is asserted.
"""
from __future__ import annotations

import argparse

from didcheck import simulate_panel, variance_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=199, help="Scheduled draws; zero disables bootstrap")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for this synthetic example")
    parser.add_argument("--plot", action="store_true", help="Show descriptive plots (requires Matplotlib)")
    args = parser.parse_args()

    panel = simulate_panel(
        n_control=200,
        n_treated=200,
        n_periods=4,
        mean_delta=0.0,
        variance_delta=0.20,
        law="normal",
        random_state=args.seed,
    )
    # "treated" is constant group membership, not treatment receipt at a date.
    # All four simulated periods belong to the common pretreatment window.
    result = variance_test(
        panel,
        outcome="outcome",
        unit="unit",
        time="time",
        treatment="treated",
        pre_periods=[1, 2, 3, 4],
        bootstrap=args.bootstrap,
        random_state=args.seed + 1,
        fisher=False,
    )
    print("Synthetic independent-unit example; no empirical application.")
    print(result.summary())
    print("Demeaned treated/control variance ratios:", result.variance_ratios)
    print("A single rejection or nonrejection does not measure calibration or power.")

    if args.plot:
        import matplotlib.pyplot as plt
        from didcheck.plotting import plot_bootstrap, plot_variance_ratios

        plot_variance_ratios(result)
        if result.bootstrap is not None:
            plot_bootstrap(result, component="sum")
        plt.show()


if __name__ == "__main__":
    main()
