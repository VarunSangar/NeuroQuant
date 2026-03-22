"""
visualization/plots.py
-----------------------
All visualization functions for NeuroQuant.

Each function accepts a DataFrame (matching TrialResult schema) and
returns a matplotlib Figure or plotly Figure. Functions are pure —
they never mutate data or have side effects beyond figure creation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


# ------------------------------------------------------------------
# Style constants
# ------------------------------------------------------------------

PALETTE = {
    "risky":    "#E63946",
    "safe":     "#457B9D",
    "optimal":  "#2A9D8F",
    "human":    "#E76F51",
    "ai":       "#264653",
    "neutral":  "#ADB5BD",
    "gain":     "#52B788",
    "loss":     "#E07A5F",
}

STYLE = {
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FFFFFF",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":      "monospace",
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ------------------------------------------------------------------
# 1. Equity Curve
# ------------------------------------------------------------------

def plot_equity_curve(
    dfs:        Dict[str, pd.DataFrame],
    title:      str = "Equity Curves",
    show_ev:    bool = True,
) -> plt.Figure:
    """
    Plot cumulative P&L for one or more agents.

    Parameters
    ----------
    dfs   : Dict mapping label → DataFrame (must have 'outcome' column)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = list(PALETTE.values())
    for idx, (label, df) in enumerate(dfs.items()):
        equity = df["outcome"].cumsum()
        color  = colors[idx % len(colors)]
        ax.plot(equity.values, label=label, color=color, linewidth=2)
        ax.fill_between(range(len(equity)), equity.values, alpha=0.08, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2. Risk-Taking Over Time
# ------------------------------------------------------------------

def plot_risk_taking_trajectory(
    df:     pd.DataFrame,
    window: int = 8,
    title:  str = "Risk-Taking Trajectory",
) -> plt.Figure:
    """
    Rolling rate of risky choices over time, with streak annotations.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: rolling risky rate
    rolling_risky = (df["choice"] == "risky").astype(float).rolling(window, min_periods=1).mean()
    rolling_opt   = df["is_optimal"].astype(float).rolling(window, min_periods=1).mean()

    ax = axes[0]
    ax.plot(rolling_risky.values, color=PALETTE["risky"],   label="Risky rate",     linewidth=2)
    ax.plot(rolling_opt.values,   color=PALETTE["optimal"], label="Optimality rate", linewidth=2, linestyle="--")
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_ylabel("Rate (rolling avg)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False)

    # Bottom panel: streak indicator
    if "current_streak" in df.columns:
        ax2 = axes[1]
        streaks = df["current_streak"].values
        colors  = [PALETTE["gain"] if s > 0 else (PALETTE["loss"] if s < 0 else PALETTE["neutral"])
                   for s in streaks]
        ax2.bar(range(len(streaks)), streaks, color=colors, alpha=0.8, width=1.0)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Streak", fontsize=10)
        ax2.set_xlabel("Trial", fontsize=11)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 3. Deviation from Optimal
# ------------------------------------------------------------------

def plot_ev_deviation(
    dfs:   Dict[str, pd.DataFrame],
    title: str = "EV Deviation from Optimal Strategy",
) -> plt.Figure:
    """
    Cumulative EV left on the table over time for multiple agents.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = list(PALETTE.values())

    # Left: cumulative EV loss
    ax = axes[0]
    for idx, (label, df) in enumerate(dfs.items()):
        dev = (df["ev_optimal"] - df["ev_chosen"]).cumsum()
        ax.plot(dev.values, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative EV Foregone ($)")
    ax.set_title("Cumulative EV Deviation", fontweight="bold")
    ax.legend(frameon=False)

    # Right: per-trial EV deviation distribution
    ax2 = axes[1]
    for idx, (label, df) in enumerate(dfs.items()):
        dev = (df["ev_optimal"] - df["ev_chosen"]).values
        ax2.hist(dev, bins=20, alpha=0.5, color=colors[idx % len(colors)],
                 label=label, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("EV Deviation per Trial ($)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of EV Deviations", fontweight="bold")
    ax2.legend(frameon=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 4. Choice Distribution
# ------------------------------------------------------------------

def plot_choice_distribution(
    dfs:   Dict[str, pd.DataFrame],
    title: str = "Choice Distribution",
) -> plt.Figure:
    """
    Stacked bar chart showing risky vs safe split per agent/condition.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    labels, risky_rates, safe_rates = [], [], []
    for label, df in dfs.items():
        r = (df["choice"] == "risky").mean()
        labels.append(label)
        risky_rates.append(r)
        safe_rates.append(1 - r)

    x    = np.arange(len(labels))
    bars = ax.bar(x, risky_rates, color=PALETTE["risky"],   label="Risky", alpha=0.85)
    ax.bar(x, safe_rates,  bottom=risky_rates, color=PALETTE["safe"], label="Safe",  alpha=0.85)

    for bar, r in zip(bars, risky_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r / 2,
                f"{r:.0%}", ha="center", va="center", color="white", fontweight="bold")

    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="50% line")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Proportion of Choices")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 5. Framing Experiment Results
# ------------------------------------------------------------------

def plot_framing_results(result) -> plt.Figure:
    """Visualize Experiment 1: Gain vs Loss Framing results."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = result.aggregate_metrics

    # Panel 1: Risky rate by frame
    ax = axes[0]
    frames = ["Gain Frame", "Loss Frame"]
    rates  = [metrics["risky_rate_gain_frame"], metrics["risky_rate_loss_frame"]]
    bars   = ax.bar(frames, rates, color=[PALETTE["gain"], PALETTE["loss"]], alpha=0.85, width=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                f"{r:.1%}", ha="center", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Proportion Risky Choices")
    ax.set_title("Risky Choice Rate by Frame", fontweight="bold")
    ax.set_ylim(0, 1)

    # Panel 2: EV deviation by frame
    ax2 = axes[1]
    devs   = [metrics["ev_deviation_gain_frame"], metrics["ev_deviation_loss_frame"]]
    bars2  = ax2.bar(frames, devs, color=[PALETTE["gain"], PALETTE["loss"]], alpha=0.85, width=0.5)
    for bar, d in zip(bars2, devs):
        ax2.text(bar.get_x() + bar.get_width() / 2, max(d + 0.5, 0.5),
                 f"${d:.2f}", ha="center", fontweight="bold")
    ax2.set_ylabel("Mean EV Left on Table ($)")
    ax2.set_title("EV Deviation by Frame", fontweight="bold")

    # Panel 3: Summary stats table
    ax3 = axes[2]
    ax3.axis("off")
    fe     = metrics["framing_effect"]
    cd     = metrics["cohens_d"]
    supported = result.hypothesis_supported
    table_data = [
        ["Metric", "Value"],
        ["Framing Effect", f"{fe:+.3f}"],
        ["Cohen's d", f"{cd:.3f}"],
        ["n Agents", str(metrics["n_agents"])],
        ["Hypothesis", "✓ SUPPORTED" if supported else "✗ NOT SUPPORTED"],
    ]
    tbl = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc="center", loc="center", bbox=[0, 0.2, 1, 0.7])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    ax3.set_title("Summary", fontweight="bold")

    fig.suptitle("Experiment 1: Gain vs Loss Framing", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 6. Streak Experiment Results
# ------------------------------------------------------------------

def plot_streak_results(result) -> plt.Figure:
    """Visualize Experiment 2: Streak-Induced Behavior results."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = result.aggregate_metrics

    # Panel 1: Risky rate by condition
    ax = axes[0]
    conditions = ["No Streak (Control)", "After Win Streak", "After Loss Streak"]
    rates      = [
        metrics["risky_rate_control"],
        metrics["risky_rate_after_win_streak"],
        metrics["risky_rate_after_loss_streak"],
    ]
    bar_colors = [PALETTE["neutral"], PALETTE["gain"], PALETTE["loss"]]
    bars = ax.bar(conditions, rates, color=bar_colors, alpha=0.85, width=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                f"{r:.1%}", ha="center", fontweight="bold")
    ax.axhline(metrics["risky_rate_control"], color="gray", linestyle="--",
               linewidth=1, label="Control baseline")
    ax.set_ylabel("Proportion Risky Choices")
    ax.set_title("Post-Streak Risk-Taking", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=10)
    ax.legend(frameon=False)

    # Panel 2: Behavioral shift
    ax2 = axes[1]
    shifts = [0, metrics["behavioral_shift_win"], metrics["behavioral_shift_loss"]]
    colors = ["gray",
              PALETTE["gain"] if shifts[1] > 0 else PALETTE["loss"],
              PALETTE["loss"] if shifts[2] < 0 else PALETTE["gain"]]
    ax2.bar(conditions, shifts, color=colors, alpha=0.85, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Behavioral Shift (Δ Risky Rate)")
    ax2.set_title("Deviation from Control Baseline", fontweight="bold")
    ax2.tick_params(axis="x", rotation=10)

    fig.suptitle("Experiment 2: Streak-Induced Behavior", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 7. Prospect Theory Curve
# ------------------------------------------------------------------

def plot_prospect_theory_curves() -> plt.Figure:
    """
    Visualize the value function and probability weighting function.
    Educational plot for the methodology section.
    """
    from psychology.prospect_theory import ProspectValueFunction, ProbabilityWeighting

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Value function
    ax = axes[0]
    x  = np.linspace(-200, 200, 400)
    vf = ProspectValueFunction()
    y  = np.array([vf(xi) for xi in x])

    ax.plot(x[x >= 0], y[x >= 0], color=PALETTE["gain"],  linewidth=2.5, label="Gains (α=0.88)")
    ax.plot(x[x <= 0], y[x <= 0], color=PALETTE["loss"],  linewidth=2.5, label="Losses (λ=2.25)")
    ax.plot(x, x * 0.3,            color="gray",           linewidth=1,   linestyle="--",  label="Linear (rational)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Outcome ($)")
    ax.set_ylabel("Subjective Value V(x)")
    ax.set_title("Prospect Theory Value Function", fontweight="bold")
    ax.legend(frameon=False)

    # Probability weighting
    ax2 = axes[1]
    p   = np.linspace(0.01, 0.99, 200)
    pw  = ProbabilityWeighting()
    wg  = np.array([pw.weight_gain(pi) for pi in p])
    wl  = np.array([pw.weight_loss(pi) for pi in p])

    ax2.plot(p, wg, color=PALETTE["gain"], linewidth=2.5, label="Gain domain (γ=0.61)")
    ax2.plot(p, wl, color=PALETTE["loss"], linewidth=2.5, label="Loss domain (δ=0.69)")
    ax2.plot(p, p,  color="gray",           linewidth=1,   linestyle="--", label="Linear (rational)")
    ax2.set_xlabel("Objective Probability p")
    ax2.set_ylabel("Decision Weight w(p)")
    ax2.set_title("Probability Weighting Function", fontweight="bold")
    ax2.legend(frameon=False)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle("Kahneman-Tversky Prospect Theory Components", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 8. Dashboard (multi-panel summary)
# ------------------------------------------------------------------

def plot_session_dashboard(df: pd.DataFrame, title: str = "Session Dashboard") -> plt.Figure:
    """
    Full 2x3 dashboard summarizing one session.
    """
    _apply_style()
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, 0])
    equity = df["outcome"].cumsum()
    ax1.plot(equity.values, color=PALETTE["risky"], linewidth=2)
    ax1.fill_between(range(len(equity)), equity.values, alpha=0.1, color=PALETTE["risky"])
    ax1.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.set_xlabel("Trial"); ax1.set_ylabel("P&L ($)")

    # 2. Optimality over time
    ax2 = fig.add_subplot(gs[0, 1])
    rolling_opt = df["is_optimal"].astype(float).rolling(8, min_periods=1).mean()
    ax2.plot(rolling_opt.values, color=PALETTE["optimal"], linewidth=2)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Rolling Optimality Rate", fontweight="bold")
    ax2.set_xlabel("Trial"); ax2.set_ylabel("Rate")

    # 3. Choice distribution
    ax3 = fig.add_subplot(gs[0, 2])
    risky_rate = (df["choice"] == "risky").mean()
    ax3.bar(["Risky", "Safe"], [risky_rate, 1 - risky_rate],
            color=[PALETTE["risky"], PALETTE["safe"]], alpha=0.85, width=0.4)
    ax3.set_ylim(0, 1)
    ax3.set_title("Choice Distribution", fontweight="bold")
    ax3.set_ylabel("Proportion")

    # 4. Cumulative EV deviation
    ax4 = fig.add_subplot(gs[1, 0])
    ev_dev = (df["ev_optimal"] - df["ev_chosen"]).cumsum()
    ax4.plot(ev_dev.values, color=PALETTE["loss"], linewidth=2)
    ax4.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax4.set_title("Cumulative EV Deviation", fontweight="bold")
    ax4.set_xlabel("Trial"); ax4.set_ylabel("EV Foregone ($)")

    # 5. Streak bar chart
    if "current_streak" in df.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        streaks = df["current_streak"].values
        bar_colors = [PALETTE["gain"] if s > 0 else (PALETTE["loss"] if s < 0 else "gray") for s in streaks]
        ax5.bar(range(len(streaks)), streaks, color=bar_colors, alpha=0.8, width=1.0)
        ax5.axhline(0, color="black", linewidth=0.5)
        ax5.set_title("Win/Loss Streaks", fontweight="bold")
        ax5.set_xlabel("Trial"); ax5.set_ylabel("Streak")

    # 6. Outcome distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df["outcome"], bins=25, color=PALETTE["ai"], alpha=0.8, edgecolor="white")
    ax6.axvline(0, color="red", linewidth=1, linestyle="--")
    ax6.axvline(df["outcome"].mean(), color="green", linewidth=1.5, linestyle="-", label=f"Mean={df['outcome'].mean():.1f}")
    ax6.set_title("Outcome Distribution", fontweight="bold")
    ax6.set_xlabel("P&L ($)"); ax6.set_ylabel("Frequency")
    ax6.legend(frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    return fig
