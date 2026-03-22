"""
psychology/biases.py
--------------------
Mathematical models of key cognitive biases observed in financial
decision-making. Each bias is implemented as a modulator that adjusts
behavior as a function of history and context.

Biases modeled:
  1. Decision Fatigue     — declining rationality over time
  2. Recency Bias         — over-weighting recent outcomes
  3. Streak Sensitivity   — behavior shifts after win/loss streaks
  4. Hot-Hand Fallacy     — expecting continuation of streaks
  5. Gambler's Fallacy    — expecting reversal of streaks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ------------------------------------------------------------------
# 1. Decision Fatigue
# ------------------------------------------------------------------

@dataclass
class DecisionFatigue:
    """
    Models cognitive depletion over sustained decision-making.

    Rationality decays as a function of trial count. The agent becomes
    increasingly likely to revert to heuristic (suboptimal) behavior.

    Parameters
    ----------
    decay_rate   : Controls how quickly rationality degrades
    floor        : Minimum rationality level (agent never becomes fully random)
    recovery     : Rate at which rationality recovers during "rest" periods
    """
    decay_rate:   float = 0.03
    floor:        float = 0.40
    recovery:     float = 0.10

    def rationality(self, trial_index: int) -> float:
        """
        Rationality ∈ [floor, 1.0] as a function of trial number.

        Uses exponential decay: r(t) = floor + (1 - floor) * exp(-decay_rate * t)
        """
        r = self.floor + (1 - self.floor) * math.exp(-self.decay_rate * trial_index)
        return float(np.clip(r, self.floor, 1.0))

    def impaired(self, trial_index: int, threshold: float = 0.7) -> bool:
        """Returns True when rationality has dropped below a threshold."""
        return self.rationality(trial_index) < threshold

    def recovery_trial(self, current_rationality: float) -> float:
        """Simulate partial recovery after a break."""
        return min(1.0, current_rationality + self.recovery)


# ------------------------------------------------------------------
# 2. Recency Bias
# ------------------------------------------------------------------

@dataclass
class RecencyBias:
    """
    Models over-weighting of recent outcomes in probability estimation.

    Agents implicitly update their beliefs about win probability based on
    recent outcomes. With high recency weight, a few recent losses
    dramatically reduces their perceived win probability even if the
    objective probability hasn't changed.

    Implemented as an exponentially-weighted moving average of outcomes.

    Parameters
    ----------
    weight       : How much to weight recent vs historical outcomes (0-1)
                   0 = full Bayesian (objective frequency), 1 = last result only
    window       : Number of trials to include
    """
    weight: float = 0.7
    window: int   = 10

    def perceived_win_prob(
        self,
        history:           list,   # List[TrialResult]
        true_probability:  float,
    ) -> float:
        """
        Estimate agent's subjective win probability given recent outcomes.

        Returns a blend of true probability and recency-weighted recent performance.
        """
        if not history:
            return true_probability

        recent  = history[-self.window:]
        recent_wins = np.array([1.0 if t.outcome > 0 else 0.0 for t in recent])

        # Exponential weights: more recent = higher weight
        weights    = np.array([self.weight ** (len(recent) - i - 1) for i in range(len(recent))])
        weights   /= weights.sum()
        recent_win_rate = float(np.dot(weights, recent_wins))

        # Blend: (1 - weight) * true + weight * recent
        perceived = (1 - self.weight) * true_probability + self.weight * recent_win_rate
        return float(np.clip(perceived, 0.0, 1.0))

    def probability_distortion(
        self,
        history:           list,
        true_probability:  float,
    ) -> float:
        """Returns perceived minus true probability (signed distortion)."""
        return self.perceived_win_prob(history, true_probability) - true_probability


# ------------------------------------------------------------------
# 3. Streak Sensitivity
# ------------------------------------------------------------------

@dataclass
class StreakSensitivity:
    """
    Models how win/loss streaks alter risk preferences.

    Two opposing effects:
    - Hot-hand fallacy: After wins, expect more wins → increase risk-taking
    - Gambler's fallacy: After wins, expect a loss → decrease risk-taking

    Which dominates depends on domain and individual. We implement both
    and allow the agent configuration to weight them.

    Parameters
    ----------
    hot_hand_weight     : How strongly winning streaks increase risk appetite (0-1)
    gamblers_weight     : How strongly winning streaks decrease risk appetite (0-1)
    streak_threshold    : Streak length before behavior changes
    sensitivity_scale   : Multiplier on effect size
    """
    hot_hand_weight:   float = 0.5
    gamblers_weight:   float = 0.5
    streak_threshold:  int   = 3
    sensitivity_scale: float = 0.2

    def __post_init__(self):
        if not np.isclose(self.hot_hand_weight + self.gamblers_weight, 1.0):
            # Normalize
            total = self.hot_hand_weight + self.gamblers_weight
            self.hot_hand_weight  /= total
            self.gamblers_weight  /= total

    def risk_adjustment(self, current_streak: int) -> float:
        """
        Returns a signed risk adjustment based on current streak.

        Positive → increased risk appetite
        Negative → decreased risk appetite

        Formula:
          adjustment = scale * streak_beyond_threshold
                       * (hot_hand_weight - gamblers_weight)
        """
        if abs(current_streak) < self.streak_threshold:
            return 0.0

        streak_length = abs(current_streak) - self.streak_threshold + 1
        direction     = np.sign(current_streak)

        # Hot-hand expects continuation (same sign as streak)
        hot_hand_effect   = direction * self.hot_hand_weight * streak_length

        # Gambler's expects reversal (opposite sign to streak)
        gamblers_effect   = -direction * self.gamblers_weight * streak_length

        raw = (hot_hand_effect + gamblers_effect) * self.sensitivity_scale
        return float(np.clip(raw, -1.0, 1.0))

    def adjusted_risky_probability(
        self,
        base_probability: float,
        current_streak:   int,
    ) -> float:
        """
        Adjust the agent's probability of choosing 'risky' based on streak.
        """
        adj = self.risk_adjustment(current_streak)
        return float(np.clip(base_probability + adj, 0.0, 1.0))


# ------------------------------------------------------------------
# 4. Composite Bias State
# ------------------------------------------------------------------

@dataclass
class BiasState:
    """
    Tracks the current magnitude of each active bias for one agent.

    This provides a single snapshot of the agent's psychological state,
    updated after every trial.
    """
    decision_fatigue:    float = 1.0   # Current rationality (1.0 = full)
    recency_distortion:  float = 0.0   # Current perceived - true probability
    streak_adjustment:   float = 0.0   # Current risk appetite adjustment
    current_streak:      int   = 0
    trial_count:         int   = 0

    def overall_rationality(self) -> float:
        """
        Composite rationality metric combining all bias sources.

        Lower = more biased behavior.
        """
        # Fatigue reduces rationality directly
        fatigue_effect = self.decision_fatigue

        # Strong recency bias reduces effective rationality
        recency_effect = 1.0 - min(abs(self.recency_distortion) * 2, 0.3)

        # Strong streak adjustment reduces rationality
        streak_effect = 1.0 - min(abs(self.streak_adjustment), 0.3)

        return float(np.clip(fatigue_effect * recency_effect * streak_effect, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "trial":              self.trial_count,
            "rationality":        round(self.overall_rationality(), 4),
            "decision_fatigue":   round(self.decision_fatigue, 4),
            "recency_distortion": round(self.recency_distortion, 4),
            "streak_adjustment":  round(self.streak_adjustment, 4),
            "current_streak":     self.current_streak,
        }


# ------------------------------------------------------------------
# Bias State Updater
# ------------------------------------------------------------------

class BiasStateUpdater:
    """
    Given a bias configuration, updates an agent's BiasState after each trial.
    """

    def __init__(
        self,
        fatigue:          DecisionFatigue  = DecisionFatigue(),
        recency:          RecencyBias      = RecencyBias(),
        streak_sens:      StreakSensitivity = StreakSensitivity(),
        true_win_prob:    float            = 0.5,
    ):
        self.fatigue       = fatigue
        self.recency       = recency
        self.streak_sens   = streak_sens
        self.true_win_prob = true_win_prob

    def update(self, history: list, current_streak: int) -> BiasState:
        """Compute current BiasState from trial history."""
        trial_count = len(history)
        return BiasState(
            decision_fatigue   = self.fatigue.rationality(trial_count),
            recency_distortion = self.recency.probability_distortion(history, self.true_win_prob),
            streak_adjustment  = self.streak_sens.risk_adjustment(current_streak),
            current_streak     = current_streak,
            trial_count        = trial_count,
        )
