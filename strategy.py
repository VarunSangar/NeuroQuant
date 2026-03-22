"""
core/strategy.py
----------------
Baseline decision strategies. Each strategy is a callable that takes a
Scenario and decision history and returns 'risky' or 'safe'.

These serve as comparison benchmarks for human behavior.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from core.scenario import Scenario


# ------------------------------------------------------------------
# Type alias (mirrors engine.py)
# ------------------------------------------------------------------

HistoryType = list  # List[TrialResult] — avoid circular import


# ------------------------------------------------------------------
# Rational Baseline
# ------------------------------------------------------------------

def rational_strategy(scenario: Scenario, history: HistoryType) -> str:
    """
    Pure EV-maximizing (risk-neutral) strategy.
    Always selects the choice with the highest expected value.
    The theoretically optimal benchmark.
    """
    return scenario.optimal_choice()


# ------------------------------------------------------------------
# Risk-Averse Baseline
# ------------------------------------------------------------------

def risk_averse_strategy(
    scenario: Scenario,
    history:  HistoryType,
    aversion: float = 0.5,
) -> str:
    """
    Mean-variance trade-off. Penalizes variance by a coefficient λ.

    Utility(choice) = EV(choice) - λ * Var(choice)

    With λ = 0.5 this strongly prefers the safe option unless the risky
    option has a substantially higher EV.
    """
    u_risky = scenario.ev("risky") - aversion * scenario.variance("risky")
    u_safe  = scenario.ev("safe")  - aversion * scenario.variance("safe")
    return "risky" if u_risky > u_safe else "safe"


def make_risk_averse(aversion: float = 0.5):
    """Factory: returns a risk-averse strategy with given λ."""
    def _strategy(scenario: Scenario, history: HistoryType) -> str:
        return risk_averse_strategy(scenario, history, aversion)
    _strategy.__name__ = f"risk_averse(λ={aversion})"
    return _strategy


# ------------------------------------------------------------------
# Risk-Seeking Baseline
# ------------------------------------------------------------------

def risk_seeking_strategy(scenario: Scenario, history: HistoryType) -> str:
    """
    Variance-preferring strategy. Selects maximum-variance option,
    breaking ties by EV.
    """
    if scenario.variance("risky") >= scenario.variance("safe"):
        return "risky"
    return "safe"


# ------------------------------------------------------------------
# Momentum Strategy (trend-following)
# ------------------------------------------------------------------

def momentum_strategy(
    scenario:      Scenario,
    history:       HistoryType,
    lookback:      int   = 5,
    threshold:     float = 0.6,
) -> str:
    """
    If the last `lookback` outcomes have been predominantly positive,
    stay risky (trend-following). Otherwise, revert to rational.

    Models a hot-hand belief system.
    """
    if len(history) < lookback:
        return rational_strategy(scenario, history)

    recent = history[-lookback:]
    win_rate = sum(1 for t in recent if t.outcome > 0) / lookback

    if win_rate >= threshold:
        return "risky"
    elif win_rate <= (1 - threshold):
        return "safe"
    return rational_strategy(scenario, history)


# ------------------------------------------------------------------
# Mean-Reversion Strategy
# ------------------------------------------------------------------

def mean_reversion_strategy(
    scenario:  Scenario,
    history:   HistoryType,
    lookback:  int = 5,
) -> str:
    """
    Gambler's fallacy: if recent outcomes have been wins, expect a loss,
    so switch to safe; after losses, expect a win, so switch to risky.

    Demonstrably suboptimal — serves as a behavioral bias benchmark.
    """
    if len(history) < lookback:
        return rational_strategy(scenario, history)

    recent   = history[-lookback:]
    win_rate = sum(1 for t in recent if t.outcome > 0) / lookback

    if win_rate > 0.6:
        return "safe"    # "Due for a loss"
    elif win_rate < 0.4:
        return "risky"   # "Due for a win"
    return rational_strategy(scenario, history)


# ------------------------------------------------------------------
# Random Baseline
# ------------------------------------------------------------------

def random_strategy(scenario: Scenario, history: HistoryType) -> str:
    """50/50 random choice. Lower bound on performance."""
    return random.choice(["risky", "safe"])


# ------------------------------------------------------------------
# Strategy Registry
# ------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "rational":        rational_strategy,
    "risk_averse":     make_risk_averse(0.5),
    "risk_seeking":    risk_seeking_strategy,
    "momentum":        momentum_strategy,
    "mean_reversion":  mean_reversion_strategy,
    "random":          random_strategy,
}


def get_strategy(name: str):
    """Look up a strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Options: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]
