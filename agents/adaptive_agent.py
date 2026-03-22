"""
agents/adaptive_agent.py
-------------------------
Adaptive heuristic AI agent that learns from its own decision history.

Unlike the behavioral agent (which is a fixed parametric model), the
adaptive agent updates its strategy based on observed outcomes. It serves
as a comparison point: how does a simple learning algorithm fare against
both the rational baseline and the behaviorally-biased human model?

Algorithm: Epsilon-Greedy + Online Expected Value Estimation
  - Maintains running EV estimates for each action per scenario type
  - Exploits current best estimate with probability (1 - ε)
  - Explores randomly with probability ε
  - ε decays over time (exploitation increases with experience)
  - Incorporates a recency-weighted update rule
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.scenario import Scenario


# ------------------------------------------------------------------
# Action value model
# ------------------------------------------------------------------

@dataclass
class ActionValueEstimate:
    """Running estimate of the value of an action."""
    n:       int   = 0      # Number of observations
    value:   float = 0.0    # Current estimate
    alpha:   float = 0.1    # Learning rate (higher = more recency)

    def update(self, reward: float) -> None:
        """Incremental mean update with optional learning rate."""
        self.n += 1
        if self.alpha is None:
            # Simple running mean
            self.value += (reward - self.value) / self.n
        else:
            # Exponential moving average
            self.value += self.alpha * (reward - self.value)


# ------------------------------------------------------------------
# Adaptive Agent
# ------------------------------------------------------------------

class AdaptiveAgent:
    """
    Epsilon-greedy adaptive agent with online EV learning.

    The agent treats each scenario (by scenario_id) as a distinct
    multi-armed bandit and learns EV estimates for risky/safe choices.

    Parameters
    ----------
    epsilon_start   : Initial exploration rate (1.0 = fully random)
    epsilon_end     : Final exploration rate (floor)
    epsilon_decay   : Exponential decay rate
    learning_rate   : EV estimate update step size
    """

    def __init__(
        self,
        epsilon_start:  float = 0.8,
        epsilon_end:    float = 0.05,
        epsilon_decay:  float = 0.05,
        learning_rate:  float = 0.15,
        seed:           Optional[int] = None,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr            = learning_rate
        self.rng           = np.random.default_rng(seed)

        # Q-table: scenario_id → {choice → ActionValueEstimate}
        self._q: Dict[str, Dict[str, ActionValueEstimate]] = defaultdict(
            lambda: {
                "risky": ActionValueEstimate(alpha=learning_rate),
                "safe":  ActionValueEstimate(alpha=learning_rate),
            }
        )
        self._step    = 0
        self._history = []

    # ------------------------------------------------------------------
    # Decision interface
    # ------------------------------------------------------------------

    def decide(self, scenario: Scenario, history: list) -> str:
        """
        Epsilon-greedy action selection.

        On exploration: random choice.
        On exploitation: choose action with highest Q-value;
            tie-break toward scenario's true optimal (rational prior).
        """
        self._history = history
        epsilon       = self._current_epsilon()

        if self.rng.random() < epsilon:
            # Explore
            return self.rng.choice(["risky", "safe"])
        else:
            # Exploit
            q_risky = self._q[scenario.scenario_id]["risky"].value
            q_safe  = self._q[scenario.scenario_id]["safe"].value

            if abs(q_risky - q_safe) < 1e-6:
                # Tie: fall back to rational prior
                return scenario.optimal_choice()
            return "risky" if q_risky > q_safe else "safe"

    def update(self, scenario: Scenario, choice: str, outcome: float) -> None:
        """Update Q-value estimate after observing an outcome."""
        self._q[scenario.scenario_id][choice].update(outcome)
        self._step += 1

    def as_strategy(self):
        """
        Return a strategy function compatible with SimulationEngine.

        Note: The engine doesn't call update() — for proper online learning,
        wrap the engine loop manually or use run_with_learning() below.
        """
        return lambda scenario, history: self.decide(scenario, history)

    # ------------------------------------------------------------------
    # Learning loop
    # ------------------------------------------------------------------

    def run_episode(
        self,
        scenarios:    List[Scenario],
        n_trials:     int,
        engine,
        rng:          Optional[np.random.Generator] = None,
    ) -> "EpisodeLog":
        """
        Run one episode with online Q-update after each trial.
        Returns a detailed episode log.
        """
        rng       = rng or np.random.default_rng()
        history   = []
        pnl       = 0.0
        choices   = []
        rewards   = []
        epsilons  = []
        optimal_n = 0

        for i in range(n_trials):
            scenario = scenarios[i % len(scenarios)]
            choice   = self.decide(scenario, history)
            outcome  = scenario.sample(choice, rng=rng)

            self.update(scenario, choice, outcome)

            pnl        += outcome
            optimal     = scenario.optimal_choice()
            is_optimal  = (choice == optimal)
            optimal_n  += int(is_optimal)

            choices.append(choice)
            rewards.append(outcome)
            epsilons.append(self._current_epsilon())

            # Minimal TrialResult-like object for history
            history.append(_MockTrial(
                trial_index    = i,
                outcome        = outcome,
                current_streak = self._compute_streak(rewards),
                choice         = choice,
                ev_chosen      = scenario.ev(choice),
                ev_optimal     = scenario.ev(optimal),
                is_optimal     = is_optimal,
            ))

        return EpisodeLog(
            choices         = choices,
            rewards         = rewards,
            epsilons        = epsilons,
            total_pnl       = pnl,
            optimality_rate = optimal_n / n_trials,
            q_snapshot      = self._q_snapshot(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_epsilon(self) -> float:
        """Exponentially decaying exploration rate."""
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              np.exp(-self.epsilon_decay * self._step)
        return float(eps)

    def _q_snapshot(self) -> dict:
        return {
            sid: {a: round(v.value, 3) for a, v in actions.items()}
            for sid, actions in self._q.items()
        }

    @staticmethod
    def _compute_streak(rewards: list) -> int:
        if not rewards:
            return 0
        streak = 0
        for r in reversed(rewards):
            if r > 0 and (streak >= 0):
                streak += 1
            elif r < 0 and (streak <= 0):
                streak -= 1
            else:
                break
        return streak

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self) -> None:
        """Reset the agent's learned Q-values and step counter."""
        self._q    = defaultdict(lambda: {
            "risky": ActionValueEstimate(alpha=self.lr),
            "safe":  ActionValueEstimate(alpha=self.lr),
        })
        self._step = 0


# ------------------------------------------------------------------
# Episode log and mock trial
# ------------------------------------------------------------------

@dataclass
class EpisodeLog:
    choices:         List[str]
    rewards:         List[float]
    epsilons:        List[float]
    total_pnl:       float
    optimality_rate: float
    q_snapshot:      dict

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({
            "trial":   range(len(self.choices)),
            "choice":  self.choices,
            "outcome": self.rewards,
            "epsilon": self.epsilons,
        })


@dataclass
class _MockTrial:
    """Lightweight trial record for agent's internal history tracking."""
    trial_index:    int
    outcome:        float
    current_streak: int
    choice:         str
    ev_chosen:      float
    ev_optimal:     float
    is_optimal:     bool
