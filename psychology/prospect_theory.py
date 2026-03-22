"""
psychology/prospect_theory.py
------------------------------
Kahneman & Tversky (1979) Prospect Theory implementation.

This module provides:
  - The value function V(x): captures diminishing sensitivity and loss aversion
  - The probability weighting function w(p): captures overweighting of small
    probabilities and underweighting of moderate-to-large probabilities
  - Prospect value computation for multi-outcome gambles

Reference:
    Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of
    decision under risk. Econometrica, 47(2), 263–291.

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory:
    Cumulative representation of uncertainty. Journal of Risk and
    Uncertainty, 5(4), 297–323.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from core.scenario import Outcome


# ------------------------------------------------------------------
# Default parameter estimates from Tversky & Kahneman (1992)
# ------------------------------------------------------------------

DEFAULT_ALPHA  = 0.88   # Diminishing sensitivity for gains
DEFAULT_BETA   = 0.88   # Diminishing sensitivity for losses
DEFAULT_LAMBDA = 2.25   # Loss aversion coefficient
DEFAULT_GAMMA  = 0.61   # Probability distortion for gains
DEFAULT_DELTA  = 0.69   # Probability distortion for losses


# ------------------------------------------------------------------
# Value Function
# ------------------------------------------------------------------

@dataclass
class ProspectValueFunction:
    """
    Kahneman-Tversky value function.

         V(x) = x^α                  for x ≥ 0  (gains)
         V(x) = -λ * (-x)^β          for x < 0  (losses)

    Key properties:
    - Concave over gains → risk aversion in gain domain
    - Convex over losses → risk seeking in loss domain
    - Steeper for losses than gains (loss aversion: λ > 1)
    - Reference point dependent (x is measured as deviation from reference)
    """
    alpha:      float = DEFAULT_ALPHA
    beta:       float = DEFAULT_BETA
    lambda_:    float = DEFAULT_LAMBDA  # Loss aversion coefficient
    reference:  float = 0.0             # Reference point (current wealth level)

    def __call__(self, x: float) -> float:
        """Evaluate V(x) at deviation x from reference point."""
        delta = x - self.reference
        if delta >= 0:
            return delta ** self.alpha
        else:
            return -self.lambda_ * ((-delta) ** self.beta)

    def vectorized(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation over an array of outcomes."""
        result = np.empty_like(x, dtype=float)
        gains  = x >= self.reference
        losses = ~gains

        result[gains]  = (x[gains] - self.reference) ** self.alpha
        result[losses] = -self.lambda_ * ((self.reference - x[losses]) ** self.beta)
        return result

    def loss_aversion_ratio(self, magnitude: float = 100.0) -> float:
        """
        Ratio of pain from loss to joy from equivalent gain.
        Should be ~2.25 with default parameters.
        """
        return abs(self(-magnitude)) / self(magnitude)


# ------------------------------------------------------------------
# Probability Weighting Function
# ------------------------------------------------------------------

@dataclass
class ProbabilityWeighting:
    """
    Tversky-Kahneman (1992) probability weighting function.

         w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)

    Properties:
    - Overweights small probabilities (w(0.01) > 0.01)
    - Underweights moderate-to-large probabilities (w(0.5) < 0.5)
    - w(0) = 0, w(1) = 1 (boundary conditions preserved)
    - Creates the classic inverse-S shaped curve
    """
    gamma: float = DEFAULT_GAMMA  # Gain domain (γ < 1 → inverse-S)
    delta: float = DEFAULT_DELTA  # Loss domain

    def weight_gain(self, p: float) -> float:
        """Weighted probability for a gain outcome."""
        if p <= 0: return 0.0
        if p >= 1: return 1.0
        g = self.gamma
        return (p ** g) / ((p ** g + (1 - p) ** g) ** (1 / g))

    def weight_loss(self, p: float) -> float:
        """Weighted probability for a loss outcome."""
        if p <= 0: return 0.0
        if p >= 1: return 1.0
        d = self.delta
        return (p ** d) / ((p ** d + (1 - p) ** d) ** (1 / d))

    def weight(self, p: float, is_gain: bool = True) -> float:
        return self.weight_gain(p) if is_gain else self.weight_loss(p)


# ------------------------------------------------------------------
# Prospect Value Calculator
# ------------------------------------------------------------------

class ProspectCalculator:
    """
    Computes the subjective value of a gamble under prospect theory.

    This is what people *feel* a gamble is worth, as opposed to its
    objective expected value.
    """

    def __init__(
        self,
        alpha:   float = DEFAULT_ALPHA,
        beta:    float = DEFAULT_BETA,
        lambda_: float = DEFAULT_LAMBDA,
        gamma:   float = DEFAULT_GAMMA,
        delta:   float = DEFAULT_DELTA,
    ):
        self.value_fn = ProspectValueFunction(
            alpha=alpha, beta=beta, lambda_=lambda_
        )
        self.weight_fn = ProbabilityWeighting(gamma=gamma, delta=delta)

    def prospect_value(self, outcomes: List[Outcome]) -> float:
        """
        Compute the overall prospect value of a multi-outcome gamble.

        PV = Σ w(p_i) * V(x_i)

        This is a simplified (non-cumulative) version adequate for
        two-outcome gambles used in most behavioral experiments.
        """
        total = 0.0
        for o in outcomes:
            is_gain = o.value >= 0
            weighted_p = self.weight_fn.weight(o.probability, is_gain=is_gain)
            total += weighted_p * self.value_fn(o.value)
        return total

    def subjective_choice(self, scenario) -> str:
        """
        Returns 'risky' or 'safe' based on prospect values rather than EV.
        Models what a psychologically-realistic agent would choose.
        """
        pv_risky = self.prospect_value(scenario.risky_outcomes)
        pv_safe  = self.prospect_value(scenario.safe_outcomes)
        return "risky" if pv_risky >= pv_safe else "safe"

    def bias_toward_rational(
        self,
        scenario,
        rationality: float = 0.5,
    ) -> str:
        """
        Interpolate between full prospect-theory choice and rational choice.

        Parameters
        ----------
        rationality : float in [0, 1]
            0 = pure prospect theory (maximum bias)
            1 = pure rational (EV-maximizing)
        """
        rational_choice = scenario.optimal_choice()
        prospect_choice = self.subjective_choice(scenario)

        if rational_choice == prospect_choice:
            return rational_choice

        # With probability `rationality`, override bias and be rational
        if np.random.random() < rationality:
            return rational_choice
        return prospect_choice

    def parameter_sensitivity(self, scenario, param: str, values: np.ndarray) -> List[dict]:
        """
        Sweep a single parameter and measure how prospect value ratio changes.
        Useful for understanding which parameters most affect choice.
        """
        results = []
        for v in values:
            kwargs = {
                "alpha": self.value_fn.alpha,
                "beta":  self.value_fn.beta,
                "lambda_": self.value_fn.lambda_,
                "gamma": self.weight_fn.gamma,
                "delta": self.weight_fn.delta,
            }
            kwargs[param] = v
            calc = ProspectCalculator(**kwargs)
            pv_risky = calc.prospect_value(scenario.risky_outcomes)
            pv_safe  = calc.prospect_value(scenario.safe_outcomes)
            results.append({
                "param_value": v,
                "pv_risky": pv_risky,
                "pv_safe": pv_safe,
                "pv_ratio": pv_risky / pv_safe if pv_safe != 0 else float("inf"),
                "choice": "risky" if pv_risky >= pv_safe else "safe",
            })
        return results
