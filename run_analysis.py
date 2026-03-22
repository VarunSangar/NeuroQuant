"""
run_analysis.py
---------------
Runs the full analysis pipeline on either a specific session CSV or
the sample dataset. Generates a comprehensive metrics report and plots.

Usage:
    python run_analysis.py                            # Analyze sample data
    python run_analysis.py --data sample_data/decisions_all.csv
    python run_analysis.py --compare                  # Compare all agent types
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np

from analysis.metrics import compute_full_metrics
from visualization.plots import (
    plot_equity_curve,
    plot_risk_taking_trajectory,
    plot_ev_deviation,
    plot_choice_distribution,
    plot_session_dashboard,
)


def load_data(path: str) -> pd.DataFrame:
    print(f"  Loading: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")
    return df


def print_metrics_table(metrics: dict, label: str = ""):
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")


def run_single_analysis(df: pd.DataFrame, output_dir: str, label: str = "session"):
    os.makedirs(output_dir, exist_ok=True)

    metrics = compute_full_metrics(df)
    print_metrics_table(metrics, label)

    with open(os.path.join(output_dir, f"{label}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Dashboard
    fig = plot_session_dashboard(df, title=f"Session Analysis: {label}")
    fig.savefig(os.path.join(output_dir, f"{label}_dashboard.png"), dpi=150, bbox_inches="tight")

    # Trajectory
    fig2 = plot_risk_taking_trajectory(df, title=f"Risk-Taking: {label}")
    fig2.savefig(os.path.join(output_dir, f"{label}_trajectory.png"), dpi=150, bbox_inches="tight")

    print(f"\n  Plots saved to {output_dir}/")


def run_comparison(df: pd.DataFrame, output_dir: str):
    """Compare all agent types in the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    agent_types = df["agent_type"].unique()
    print(f"\n  Comparing {len(agent_types)} agent types: {list(agent_types)}")

    dfs_by_type = {t: df[df["agent_type"] == t] for t in agent_types}

    # All metrics
    all_metrics = {}
    for t, sub_df in dfs_by_type.items():
        all_metrics[t] = compute_full_metrics(sub_df)

    # Print comparison table
    print(f"\n{'═'*75}")
    print(f"  AGENT COMPARISON")
    print(f"{'═'*75}")
    header = f"  {'Agent':<22} {'Optimality':>10} {'Risky%':>8} {'EV Dev':>8} {'Sharpe':>8} {'Drawdown':>10}"
    print(header)
    print(f"  {'─'*71}")
    for t, m in all_metrics.items():
        print(
            f"  {t:<22} {m['optimality_rate']:>10.1%} "
            f"{m['risky_choice_rate']:>8.1%} "
            f"{m['ev_deviation']:>8.3f} "
            f"{m['sharpe_ratio']:>8.3f} "
            f"{m['max_drawdown']:>10.2f}"
        )
    print(f"{'═'*75}")

    # Save comparison
    with open(os.path.join(output_dir, "agent_comparison.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Equity curves
    fig = plot_equity_curve(dfs_by_type, title="Equity Curves by Agent Type")
    fig.savefig(os.path.join(output_dir, "comparison_equity.png"), dpi=150, bbox_inches="tight")

    # EV deviation comparison
    fig2 = plot_ev_deviation(dfs_by_type, title="EV Deviation by Agent Type")
    fig2.savefig(os.path.join(output_dir, "comparison_ev_deviation.png"), dpi=150, bbox_inches="tight")

    # Choice distribution
    fig3 = plot_choice_distribution(dfs_by_type, title="Choice Distribution by Agent Type")
    fig3.savefig(os.path.join(output_dir, "comparison_choices.png"), dpi=150, bbox_inches="tight")

    print(f"\n  Comparison plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="NeuroQuant: Analysis Runner")
    parser.add_argument("--data",    default="sample_data/decisions_all.csv",
                        help="Path to decisions CSV")
    parser.add_argument("--output",  default="results/analysis",
                        help="Output directory for plots and metrics")
    parser.add_argument("--compare", action="store_true",
                        help="Compare agent types (requires agent_type column)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroQuant: Analysis Runner")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"\n  Data file not found: {args.data}")
        print("  Run: python sample_data/generate_sample.py  first\n")
        sys.exit(1)

    df = load_data(args.data)

    if args.compare and "agent_type" in df.columns:
        run_comparison(df, args.output)
    else:
        run_single_analysis(df, args.output, label="full_dataset")

    print("\n  Analysis complete.")


if __name__ == "__main__":
    main()
