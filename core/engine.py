"""
core/engine.py
--------------
Monte Carlo simulation engine. Runs sequences of scenarios against any
agent strategy and collects realized outcomes, streaks, and equity paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.scenario import Scenario


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TrialResult:
    """Single realized trial within a simulation run."""
    trial_index:   int
    scenario_id:   str
    choice:        str          # 'risky' or 'safe'
    outcome:       float        # Realized P&L
    ev_chosen:     float        # EV of chosen option
    ev_optimal:    float        # EV of optimal option
    optimal_choice: str
    is_optimal:    bool
    cumulative_pnl: float
    current_streak: int         # Positive = win streak, negative = loss streak
    frame:         str


@dataclass
class SimulationResult:
    """Full result of one simulation run."""
    trials:          List[TrialResult]
    total_pnl:       float
    optimal_pnl:     float      # What a rational agent would have accumulated
    ev_deviation:    float      # Average EV left on the table per trial
    optimality_rate: float      # Fraction of trials where optimal choice was made
    max_win_streak:  int
    max_loss_streak: int
    metadata:        dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [vars(t) for t in self.trials]
        return pd.DataFrame(rows)

    @property
    def sharpe(self) -> float:
        pnls = np.array([t.outcome for t in self.trials])
        return float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        equity = np.cumsum([t.outcome for t in self.trials])
        peak   = np.maximum.accumulate(equity)
        dd     = equity - peak
        return float(dd.min())


# ------------------------------------------------------------------
# Strategy type alias
# A strategy is a callable: (scenario, history) -> 'risky' | 'safe'
# ------------------------------------------------------------------

StrategyFn = Callable[[Scenario, List[TrialResult]], str]


# ------------------------------------------------------------------
# Simulation Engine
# ------------------------------------------------------------------

class SimulationEngine:
    """
    Runs Monte Carlo sequences of scenarios against a strategy function.

    The engine is strategy-agnostic — it accepts any callable that
    maps (Scenario, history) → choice string. This allows plugging in
    human agents, rule-based agents, or AI agents uniformly.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        scenarios:    List[Scenario],
        strategy:     StrategyFn,
        n_trials:     Optional[int] = None,
        initial_cash: float = 1000.0,
    ) -> SimulationResult:
        """
        Run one simulation pass.

        Parameters
        ----------
        scenarios    : Ordered or random pool of scenarios to present
        strategy     : Decision function (scenario, history) -> choice
        n_trials     : If set, cycles through scenarios for this many trials
        initial_cash : Starting capital (affects psychology but not core EV math)
        """
        if n_trials is not None:
            # Cycle through the scenario list as needed
            n = n_trials
            scenario_seq = [scenarios[i % len(scenarios)] for i in range(n)]
        else:
            scenario_seq = scenarios
            n = len(scenarios)

        history:       List[TrialResult] = []
        cumulative_pnl = 0.0
        optimal_pnl    = 0.0
        streak         = 0

        for i, scenario in enumerate(scenario_seq):
            choice        = strategy(scenario, history)
            outcome       = scenario.sample(choice, rng=self.rng)
            ev_chosen     = scenario.ev(choice)
            optimal       = scenario.optimal_choice()
            ev_optimal    = scenario.ev(optimal)
            is_optimal    = (choice == optimal)

            cumulative_pnl += outcome
            optimal_pnl    += scenario.sample(optimal, rng=self.rng)

            # Track win/loss streak
            if outcome > 0:
                streak = max(streak + 1, 1)
            elif outcome < 0:
                streak = min(streak - 1, -1)
            else:
                streak = 0

            trial = TrialResult(
                trial_index    = i,
                scenario_id    = scenario.scenario_id,
                choice         = choice,
                outcome        = outcome,
                ev_chosen      = ev_chosen,
                ev_optimal     = ev_optimal,
                optimal_choice = optimal,
                is_optimal     = is_optimal,
                cumulative_pnl = cumulative_pnl,
                current_streak = streak,
                frame          = scenario.frame.value,
            )
            history.append(trial)

        # Aggregate metrics
        ev_gaps = [t.ev_optimal - t.ev_chosen for t in history]
        streaks = [t.current_streak for t in history]

        return SimulationResult(
            trials          = history,
            total_pnl       = cumulative_pnl,
            optimal_pnl     = optimal_pnl,
            ev_deviation    = float(np.mean(ev_gaps)),
            optimality_rate = float(np.mean([t.is_optimal for t in history])),
            max_win_streak  = int(max(streaks)),
            max_loss_streak = int(min(streaks)),
        )

    # ------------------------------------------------------------------
    # Monte Carlo ensemble
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        scenarios:  List[Scenario],
        strategy:   StrategyFn,
        n_runs:     int   = 1000,
        n_trials:   int   = 50,
    ) -> pd.DataFrame:
        """
        Run n_runs independent simulations and return summary statistics.

        Returns a DataFrame with one row per run containing key metrics.
        """
        records = []
        for run_idx in range(n_runs):
            result = self.run(scenarios, strategy, n_trials=n_trials)
            records.append({
                "run":             run_idx,
                "total_pnl":       result.total_pnl,
                "optimal_pnl":     result.optimal_pnl,
                "ev_deviation":    result.ev_deviation,
                "optimality_rate": result.optimality_rate,
                "sharpe":          result.sharpe,
                "max_drawdown":    result.max_drawdown,
                "max_win_streak":  result.max_win_streak,
                "max_loss_streak": result.max_loss_streak,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Streak injection (for Experiment 2)
    # ------------------------------------------------------------------

    def inject_streak(
        self,
        scenario:    Scenario,
        streak_type: str,        # 'win' or 'loss'
        length:      int,
        strategy:    StrategyFn,
    ) -> Tuple[List[TrialResult], SimulationResult]:
        """
        Force a controlled streak by overriding outcome sampling,
        then run post-streak trials using the normal strategy.

        Returns (streak_trials, post_streak_result).
        """
        forced_outcomes = {
            "win":  max(o.value for o in scenario.risky_outcomes),
            "loss": min(o.value for o in scenario.risky_outcomes),
        }
        forced_value = forced_outcomes[streak_type]

        streak_history: List[TrialResult] = []
        cumulative_pnl = 0.0
        streak_counter = 0

        for i in range(length):
            choice         = strategy(scenario, streak_history)
            outcome        = forced_value          # Force outcome
            cumulative_pnl += outcome
            streak_counter  = i + 1 if streak_type == "win" else -(i + 1)

            streak_history.append(TrialResult(
                trial_index    = i,
                scenario_id    = scenario.scenario_id,
                choice         = choice,
                outcome        = outcome,
                ev_chosen      = scenario.ev(choice),
                ev_optimal     = scenario.ev(scenario.optimal_choice()),
                optimal_choice = scenario.optimal_choice(),
                is_optimal     = (choice == scenario.optimal_choice()),
                cumulative_pnl = cumulative_pnl,
                current_streak = streak_counter,
                frame          = scenario.frame.value,
            ))

        # Now run post-streak trials
        post_result = self.run(
            scenarios=[scenario] * 20,
            strategy=lambda s, h: strategy(s, streak_history + h),
            n_trials=20,
        )

        return streak_history, post_result
