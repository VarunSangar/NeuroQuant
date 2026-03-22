"""
run_experiment.py
-----------------
Command-line runner for NeuroQuant experiments.

Usage:
    python run_experiment.py --experiment framing --simulations 200
    python run_experiment.py --experiment streaks --simulations 200
    python run_experiment.py --experiment both    --simulations 100

Outputs results to results/ directory and prints a summary.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI

# Rich for pretty terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from experiments.exp1_framing import make_framing_experiment
from experiments.exp2_streaks import make_streak_experiment
from visualization.plots import (
    plot_framing_results,
    plot_streak_results,
    plot_prospect_theory_curves,
)

console = Console() if HAS_RICH else None


def run_framing(args):
    print("\n[1/3] Running Experiment 1: Gain vs Loss Framing...")
    exp    = make_framing_experiment(
        n_trials      = args.trials,
        n_simulations = args.simulations,
        seed          = args.seed,
    )
    result = exp.analyze()
    print(result.summary())

    # Save results
    os.makedirs("results", exist_ok=True)
    result.to_json("results/exp1_framing_results.json")

    # Save plots
    fig = plot_framing_results(result)
    fig.savefig("results/exp1_framing_plot.png", dpi=150, bbox_inches="tight")
    print("  → Results saved to results/exp1_framing_results.json")
    print("  → Plot saved to results/exp1_framing_plot.png")
    return result


def run_streaks(args):
    print("\n[2/3] Running Experiment 2: Streak-Induced Behavior...")
    exp    = make_streak_experiment(
        n_trials      = args.trials,
        n_simulations = args.simulations,
        seed          = args.seed,
    )
    result = exp.analyze()
    print(result.summary())

    os.makedirs("results", exist_ok=True)
    result.to_json("results/exp2_streaks_results.json")

    fig = plot_streak_results(result)
    fig.savefig("results/exp2_streaks_plot.png", dpi=150, bbox_inches="tight")
    print("  → Results saved to results/exp2_streaks_results.json")
    print("  → Plot saved to results/exp2_streaks_plot.png")
    return result


def run_prospect_theory_plot():
    print("\n[3/3] Generating Prospect Theory reference plots...")
    os.makedirs("results", exist_ok=True)
    fig = plot_prospect_theory_curves()
    fig.savefig("results/prospect_theory_curves.png", dpi=150, bbox_inches="tight")
    print("  → Plot saved to results/prospect_theory_curves.png")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroQuant: Run behavioral decision-making experiments"
    )
    parser.add_argument(
        "--experiment", choices=["framing", "streaks", "both"], default="both",
        help="Which experiment to run (default: both)"
    )
    parser.add_argument(
        "--trials", type=int, default=40,
        help="Number of trials per simulated participant (default: 40)"
    )
    parser.add_argument(
        "--simulations", type=int, default=200,
        help="Number of simulated participants per condition (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroQuant: Decision-Making Under Uncertainty")
    print("=" * 60)
    print(f"  Mode:        {args.experiment}")
    print(f"  Simulations: {args.simulations} agents per condition")
    print(f"  Trials:      {args.trials} per agent")
    print(f"  Seed:        {args.seed}")
    print("=" * 60)

    if args.experiment in ("framing", "both"):
        run_framing(args)

    if args.experiment in ("streaks", "both"):
        run_streaks(args)

    run_prospect_theory_plot()

    print("\n" + "=" * 60)
    print("  All experiments complete. Results in results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
