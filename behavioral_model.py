"""
psychology/behavioral_model.py
-------------------------------
Composite behavioral agent model.

Combines prospect theory, decision fatigue, recency bias, and streak
sensitivity into a single parameterized decision-making model that
simulates a human participant in the trading experiments.

The key abstraction is the BehavioralAgent: a configurable entity whose
decisions are driven by psychological parameters rather than EV optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from psychology.prospect_theory import (
    ProspectCalculator,
    DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_LAMBDA, DEFAULT_GAMMA, DEFAULT_DELTA,
)
from psychology.biases import (
    DecisionFatigue, RecencyBias, StreakSensitivity, BiasStateUpdater, BiasState,
)
from core.scenario import Scenario


# ------------------------------------------------------------------
# Behavioral Profile Presets
# ------------------------------------------------------------------

@dataclass
class BehavioralProfile:
    """
    A named configuration of psychological parameters.

    Used to define population archetypes (e.g., "risk-averse investor",
    "overconfident trader") for comparative simulations.
    """
    name:                str

    # Prospect theory parameters
    alpha:               float = DEFAULT_ALPHA
    beta:                float = DEFAULT_BETA
    lambda_:             float = DEFAULT_LAMBDA
    gamma:               float = DEFAULT_GAMMA
    delta:               float = DEFAULT_DELTA

    # Bias parameters
    fatigue_decay:       float = 0.03
    fatigue_floor:       float = 0.40
    recency_weight:      float = 0.70
    hot_hand_weight:     float = 0.50
    gamblers_weight:     float = 0.50
    streak_threshold:    int   = 3

    # Base rationality (independent noise)
    base_rationality:    float = 0.75

    @classmethod
    def rational_baseline(cls) -> "BehavioralProfile":
        """An EV-maximizing agent with no psychological biases."""
        return cls(
            name="rational_baseline",
            alpha=1.0, beta=1.0, lambda_=1.0, gamma=1.0, delta=1.0,
            fatigue_decay=0.0, fatigue_floor=1.0,
            recency_weight=0.0,
            hot_hand_weight=0.5, gamblers_weight=0.5,
            base_rationality=1.0,
        )

    @classmethod
    def loss_averse_investor(cls) -> "BehavioralProfile":
        """Classic loss-averse retail investor."""
        return cls(
            name="loss_averse_investor",
            lambda_=3.0, alpha=0.80, beta=0.85,
            recency_weight=0.6,
            base_rationality=0.65,
        )

    @classmethod
    def overconfident_trader(cls) -> "BehavioralProfile":
        """Overconfident, trend-following trader."""
        return cls(
            name="overconfident_trader",
            hot_hand_weight=0.85, gamblers_weight=0.15,
            streak_threshold=2,
            recency_weight=0.80,
            base_rationality=0.55,
        )

    @classmethod
    def fatigued_analyst(cls) -> "BehavioralProfile":
        """Decision-fatigued participant — degrades quickly."""
        return cls(
            name="fatigued_analyst",
            fatigue_decay=0.08, fatigue_floor=0.25,
            base_rationality=0.70,
        )


# ------------------------------------------------------------------
# Behavioral Agent
# ------------------------------------------------------------------

class BehavioralAgent:
    """
    A psychologically realistic decision-making agent.

    Combines prospect theory value computation with dynamic bias state
    tracking to produce human-like choices in trading scenarios.

    The agent's decision process:
      1. Compute prospect values for risky/safe using value + weight functions
      2. Determine prospect-theory preferred choice
      3. Apply current rationality modifier (based on fatigue + biases)
      4. With probability (1 - rationality), deviate to the biased choice
      5. Update internal state after observing outcome
    """

    def __init__(self, profile: Optional[BehavioralProfile] = None, seed: Optional[int] = None):
        self.profile  = profile or BehavioralProfile("default")
        self.rng      = np.random.default_rng(seed)
        self._history = []  # Internal reference to trial results
        self._streak  = 0

        # Initialize sub-components from profile
        self.prospect  = ProspectCalculator(
            alpha   = self.profile.alpha,
            beta    = self.profile.beta,
            lambda_ = self.profile.lambda_,
            gamma   = self.profile.gamma,
            delta   = self.profile.delta,
        )
        self._bias_updater = BiasStateUpdater(
            fatigue      = DecisionFatigue(
                decay_rate = self.profile.fatigue_decay,
                floor      = self.profile.fatigue_floor,
            ),
            recency      = RecencyBias(weight=self.profile.recency_weight),
            streak_sens  = StreakSensitivity(
                hot_hand_weight  = self.profile.hot_hand_weight,
                gamblers_weight  = self.profile.gamblers_weight,
                streak_threshold = self.profile.streak_threshold,
            ),
        )

    # ------------------------------------------------------------------
    # Decision interface (compatible with StrategyFn)
    # ------------------------------------------------------------------

    def decide(self, scenario: Scenario, history: list) -> str:
        """
        Make a decision given the current scenario and trial history.

        Compatible with the StrategyFn type alias used by SimulationEngine.
        """
        self._history = history
        if history:
            self._streak = history[-1].current_streak

        bias_state    = self._current_bias_state()
        rationality   = bias_state.overall_rationality() * self.profile.base_rationality
        rationality   = float(np.clip(rationality, 0.0, 1.0))

        # Get the "biased" choice from prospect theory
        biased_choice  = self.prospect.subjective_choice(scenario)
        rational_choice = scenario.optimal_choice()

        # With probability `rationality`, choose rationally
        if self.rng.random() < rationality:
            return rational_choice
        else:
            return biased_choice

    def get_bias_state(self, history: list) -> BiasState:
        """Return current bias state snapshot (for analysis/logging)."""
        streak = history[-1].current_streak if history else 0
        return self._bias_updater.update(history, streak)

    def _current_bias_state(self) -> BiasState:
        streak = self._history[-1].current_streak if self._history else 0
        return self._bias_updater.update(self._history, streak)

    # ------------------------------------------------------------------
    # Convenience: return callable for SimulationEngine
    # ------------------------------------------------------------------

    def as_strategy(self):
        """Return a strategy function bound to this agent."""
        return lambda scenario, history: self.decide(scenario, history)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def rationality_trajectory(self, n_trials: int) -> List[float]:
        """
        Project rationality over time given fatigue parameters alone.
        Useful for visualizing expected degradation curve.
        """
        return [
            self._bias_updater.fatigue.rationality(t) * self.profile.base_rationality
            for t in range(n_trials)
        ]


# ------------------------------------------------------------------
# Population simulation
# ------------------------------------------------------------------

def simulate_population(
    profile:   BehavioralProfile,
    scenario,
    n_agents:  int = 100,
    seed:      Optional[int] = 42,
) -> List[str]:
    """
    Simulate choices from a population of agents sharing the same profile.

    Returns list of choices ('risky'/'safe') for statistical analysis.
    """
    rng = np.random.default_rng(seed)
    choices = []
    for i in range(n_agents):
        agent  = BehavioralAgent(profile=profile, seed=int(rng.integers(0, 99999)))
        choice = agent.decide(scenario, [])
        choices.append(choice)
    return choices
