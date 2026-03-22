"""
core/scenario.py
----------------
Defines trading scenarios with configurable probability distributions,
expected values, and framing. A Scenario is the atomic unit of the
simulation — a single decision presented to an agent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class Frame(Enum):
    """How a scenario is presented to the decision-maker."""
    NEUTRAL = "neutral"
    GAIN    = "gain"    # Emphasizes potential gains
    LOSS    = "loss"    # Emphasizes potential losses


class RiskProfile(Enum):
    """Rough classification of scenario risk."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class Outcome:
    """A single possible outcome in a scenario."""
    value: float       # Dollar P&L
    probability: float # Probability of this outcome occurring

    def __post_init__(self):
        if not (0 <= self.probability <= 1):
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")


@dataclass
class Scenario:
    """
    A trading decision scenario.

    Contains two choices: a 'risky' option (higher variance) and a
    'safe' option (lower variance). Both choices may have positive or
    negative expected value depending on the experimental design.

    Parameters
    ----------
    scenario_id   : Unique identifier
    description   : Human-readable description shown in the UI
    frame         : Gain or loss framing applied to presentation
    risky_outcomes: List of (value, probability) outcomes for risky choice
    safe_outcomes : List of (value, probability) outcomes for safe choice
    """
    description:    str
    risky_outcomes: List[Outcome]
    safe_outcomes:  List[Outcome]
    frame:          Frame = Frame.NEUTRAL
    scenario_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata:       dict  = field(default_factory=dict)

    def __post_init__(self):
        self._validate_outcomes(self.risky_outcomes, "risky")
        self._validate_outcomes(self.safe_outcomes,  "safe")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_outcomes(outcomes: List[Outcome], label: str) -> None:
        total_prob = sum(o.probability for o in outcomes)
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(
                f"Probabilities for '{label}' outcomes must sum to 1.0, got {total_prob:.4f}"
            )

    # ------------------------------------------------------------------
    # Expected Value & Risk Metrics
    # ------------------------------------------------------------------

    def ev(self, choice: str) -> float:
        """Expected value of 'risky' or 'safe' choice."""
        outcomes = self.risky_outcomes if choice == "risky" else self.safe_outcomes
        return sum(o.value * o.probability for o in outcomes)

    def variance(self, choice: str) -> float:
        """Variance of the chosen option."""
        outcomes = self.risky_outcomes if choice == "risky" else self.safe_outcomes
        mean = self.ev(choice)
        return sum(o.probability * (o.value - mean) ** 2 for o in outcomes)

    def std(self, choice: str) -> float:
        return float(np.sqrt(self.variance(choice)))

    def sharpe_ratio(self, choice: str, risk_free: float = 0.0) -> float:
        """Simplified Sharpe ratio for a scenario choice."""
        s = self.std(choice)
        return (self.ev(choice) - risk_free) / s if s > 0 else 0.0

    def optimal_choice(self) -> str:
        """Returns 'risky' or 'safe' based purely on expected value."""
        return "risky" if self.ev("risky") >= self.ev("safe") else "safe"

    def risk_profile(self) -> RiskProfile:
        risky_std = self.std("risky")
        if risky_std < 20:
            return RiskProfile.LOW
        elif risky_std < 60:
            return RiskProfile.MEDIUM
        else:
            return RiskProfile.HIGH

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, choice: str, rng: Optional[np.random.Generator] = None) -> float:
        """Draw a realized P&L for the given choice."""
        rng = rng or np.random.default_rng()
        outcomes = self.risky_outcomes if choice == "risky" else self.safe_outcomes
        values = [o.value for o in outcomes]
        probs  = [o.probability for o in outcomes]
        return float(rng.choice(values, p=probs))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def framed_description(self) -> str:
        """Return description with gain/loss framing applied."""
        risky_ev  = self.ev("risky")
        safe_ev   = self.ev("safe")
        risky_std = self.std("risky")

        if self.frame == Frame.GAIN:
            return (
                f"[OPPORTUNITY] {self.description}\n"
                f"  Option A (Risky):  Expected gain ${risky_ev:+.2f}  (σ=${risky_std:.2f})\n"
                f"  Option B (Safe):   Guaranteed gain ${safe_ev:+.2f}"
            )
        elif self.frame == Frame.LOSS:
            return (
                f"[RISK ALERT] {self.description}\n"
                f"  Option A (Accept risk): Expected loss ${abs(risky_ev):.2f}  (σ=${risky_std:.2f})\n"
                f"  Option B (Accept loss): Certain loss  ${abs(safe_ev):.2f}"
            )
        else:
            return (
                f"{self.description}\n"
                f"  Option A (Risky):  EV=${risky_ev:+.2f}, σ=${risky_std:.2f}\n"
                f"  Option B (Safe):   EV=${safe_ev:+.2f}, σ=${self.std('safe'):.2f}"
            )

    def summary_dict(self) -> dict:
        return {
            "id":           self.scenario_id,
            "description":  self.description,
            "frame":        self.frame.value,
            "risky_ev":     round(self.ev("risky"), 4),
            "safe_ev":      round(self.ev("safe"), 4),
            "risky_std":    round(self.std("risky"), 4),
            "safe_std":     round(self.std("safe"), 4),
            "optimal":      self.optimal_choice(),
            "risk_profile": self.risk_profile().value,
        }


# ------------------------------------------------------------------
# Scenario Library
# ------------------------------------------------------------------

class ScenarioLibrary:
    """
    Pre-built scenario collections for experiments.
    All EVs are stated from a neutral perspective.
    """

    @staticmethod
    def classic_coin_flip(frame: Frame = Frame.NEUTRAL) -> Scenario:
        """50/50 coin flip — risky dominates by EV."""
        return Scenario(
            description="A coin is flipped. Heads you win, tails you lose.",
            risky_outcomes=[Outcome(100, 0.5), Outcome(-20, 0.5)],
            safe_outcomes= [Outcome(30,  1.0)],
            frame=frame,
        )

    @staticmethod
    def asymmetric_bet(frame: Frame = Frame.NEUTRAL) -> Scenario:
        """High-upside low-probability bet vs safe moderate gain."""
        return Scenario(
            description="High-risk trade: small chance of large gain.",
            risky_outcomes=[Outcome(500, 0.15), Outcome(-50, 0.85)],
            safe_outcomes= [Outcome(20,  1.0)],
            frame=frame,
        )

    @staticmethod
    def loss_recovery(frame: Frame = Frame.NEUTRAL) -> Scenario:
        """Post-loss scenario — tests gambler's fallacy and loss-chasing."""
        return Scenario(
            description="After recent losses, a recovery trade is available.",
            risky_outcomes=[Outcome(200, 0.3), Outcome(-150, 0.7)],
            safe_outcomes= [Outcome(-30, 1.0)],
            frame=frame,
        )

    @staticmethod
    def near_certain_gain(frame: Frame = Frame.NEUTRAL) -> Scenario:
        """Near-certain gain vs sure gain — tests risk sensitivity near certainty."""
        return Scenario(
            description="Near-certain positive trade vs locked-in profit.",
            risky_outcomes=[Outcome(100, 0.95), Outcome(-200, 0.05)],
            safe_outcomes= [Outcome(80,  1.0)],
            frame=frame,
        )

    @staticmethod
    def negative_ev_gamble(frame: Frame = Frame.NEUTRAL) -> Scenario:
        """Negative EV risky option — should always choose safe."""
        return Scenario(
            description="A tempting but statistically unfavorable trade.",
            risky_outcomes=[Outcome(300, 0.2), Outcome(-100, 0.8)],
            safe_outcomes= [Outcome(10,  1.0)],
            frame=frame,
        )

    @classmethod
    def framing_pair(cls) -> Tuple[Scenario, Scenario]:
        """
        Returns identical scenarios under gain and loss framing.
        Used in Experiment 1.
        """
        return (
            cls.classic_coin_flip(Frame.GAIN),
            cls.classic_coin_flip(Frame.LOSS),
        )

    @classmethod
    def standard_battery(cls, frame: Frame = Frame.NEUTRAL) -> List[Scenario]:
        """A balanced set of scenarios covering different risk profiles."""
        return [
            cls.classic_coin_flip(frame),
            cls.asymmetric_bet(frame),
            cls.near_certain_gain(frame),
            cls.negative_ev_gamble(frame),
            cls.loss_recovery(frame),
        ]
