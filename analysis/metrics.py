"""
analysis/metrics.py
--------------------
Behavioral and financial metrics computed from simulation/experiment data.

All functions accept a pandas DataFrame with columns matching TrialResult
field names, or an ExperimentResult object.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------
# Core behavioral metrics
# ------------------------------------------------------------------

def ev_deviation(df: pd.DataFrame) -> float:
    """Mean EV left on the table per trial (ev_optimal - ev_chosen)."""
    return float((df["ev_optimal"] - df["ev_chosen"]).mean())


def cumulative_ev_loss(df: pd.DataFrame) -> pd.Series:
    """Running sum of EV foregone over trials."""
    return (df["ev_optimal"] - df["ev_chosen"]).cumsum()


def optimality_rate(df: pd.DataFrame) -> float:
    """Fraction of trials where the optimal choice was made."""
    return float(df["is_optimal"].mean())


def risky_choice_rate(df: pd.DataFrame) -> float:
    return float((df["choice"] == "risky").mean())


def rolling_optimality(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Rolling optimality rate — reveals temporal trends in decision quality."""
    return df["is_optimal"].astype(float).rolling(window, min_periods=1).mean()


def rolling_risky_rate(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return (df["choice"] == "risky").astype(float).rolling(window, min_periods=1).mean()


# ------------------------------------------------------------------
# Financial metrics
# ------------------------------------------------------------------

def sharpe_ratio(df: pd.DataFrame, risk_free: float = 0.0) -> float:
    pnl = df["outcome"]
    std = pnl.std()
    return float((pnl.mean() - risk_free) / std) if std > 0 else 0.0


def max_drawdown(df: pd.DataFrame) -> float:
    equity = df["outcome"].cumsum()
    peak   = equity.cummax()
    return float((equity - peak).min())


def calmar_ratio(df: pd.DataFrame) -> float:
    """Annualized-equivalent return / max drawdown."""
    total_pnl = df["outcome"].sum()
    mdd       = abs(max_drawdown(df))
    return float(total_pnl / mdd) if mdd > 0 else 0.0


def win_rate(df: pd.DataFrame) -> float:
    return float((df["outcome"] > 0).mean())


def profit_factor(df: pd.DataFrame) -> float:
    """Gross profit / gross loss."""
    gross_profit = df[df["outcome"] > 0]["outcome"].sum()
    gross_loss   = abs(df[df["outcome"] < 0]["outcome"].sum())
    return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")


# ------------------------------------------------------------------
# Streak analysis
# ------------------------------------------------------------------

def streak_risk_correlation(df: pd.DataFrame) -> float:
    """
    Pearson correlation between current streak and probability of choosing risky.
    A positive correlation indicates hot-hand behavior.
    A negative correlation indicates gambler's fallacy.
    """
    if "current_streak" not in df.columns:
        return float("nan")
    risky_binary = (df["choice"] == "risky").astype(float)
    corr, _      = stats.pearsonr(df["current_streak"], risky_binary)
    return float(corr)


def post_streak_deviation(
    df: pd.DataFrame,
    streak_threshold: int = 3,
    window: int = 5,
) -> Dict[str, float]:
    """
    Measure EV deviation in the window immediately following a streak.

    Returns dict with 'post_win_deviation' and 'post_loss_deviation'.
    """
    results = {"post_win_deviation": [], "post_loss_deviation": []}

    for i in range(len(df) - window):
        streak = df.iloc[i]["current_streak"]
        if streak >= streak_threshold:
            post_slice = df.iloc[i + 1: i + 1 + window]
            results["post_win_deviation"].append(
                (post_slice["ev_optimal"] - post_slice["ev_chosen"]).mean()
            )
        elif streak <= -streak_threshold:
            post_slice = df.iloc[i + 1: i + 1 + window]
            results["post_loss_deviation"].append(
                (post_slice["ev_optimal"] - post_slice["ev_chosen"]).mean()
            )

    return {
        "post_win_deviation":  float(np.mean(results["post_win_deviation"]))  if results["post_win_deviation"]  else float("nan"),
        "post_loss_deviation": float(np.mean(results["post_loss_deviation"])) if results["post_loss_deviation"] else float("nan"),
    }


# ------------------------------------------------------------------
# Temporal bias metrics
# ------------------------------------------------------------------

def rationality_decay_slope(df: pd.DataFrame) -> float:
    """
    Linear regression slope of optimality_rate over trial index.
    Negative slope → rationality decays over time (decision fatigue).
    """
    x = np.arange(len(df))
    y = df["is_optimal"].astype(float).values
    slope, _, r, _, _ = stats.linregress(x, y)
    return float(slope)


def recency_bias_index(df: pd.DataFrame, lookback: int = 5) -> float:
    """
    Measures how much recent outcomes predict next choice (above base rate).
    Higher value → stronger recency bias.
    """
    if len(df) < lookback + 1:
        return 0.0

    predictions = []
    for i in range(lookback, len(df)):
        recent_wins = (df.iloc[i - lookback: i]["outcome"] > 0).mean()
        predicted_risky = 1.0 if recent_wins > 0.5 else 0.0
        actual_risky    = 1.0 if df.iloc[i]["choice"] == "risky" else 0.0
        predictions.append(predicted_risky == actual_risky)

    return float(np.mean(predictions))


# ------------------------------------------------------------------
# Full summary report
# ------------------------------------------------------------------

def compute_full_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute all metrics and return as a flat dict."""
    return {
        # Behavioral
        "ev_deviation":            round(ev_deviation(df),            4),
        "optimality_rate":         round(optimality_rate(df),         4),
        "risky_choice_rate":       round(risky_choice_rate(df),       4),
        "rationality_decay_slope": round(rationality_decay_slope(df), 6),
        "recency_bias_index":      round(recency_bias_index(df),      4),
        "streak_risk_correlation": round(streak_risk_correlation(df), 4),
        # Financial
        "total_pnl":               round(df["outcome"].sum(),         2),
        "win_rate":                round(win_rate(df),                4),
        "sharpe_ratio":            round(sharpe_ratio(df),            4),
        "max_drawdown":            round(max_drawdown(df),            2),
        "profit_factor":           round(profit_factor(df),           4),
        # Streak
        **{k: round(v, 4) for k, v in post_streak_deviation(df).items()
           if not np.isnan(v)},
    }
