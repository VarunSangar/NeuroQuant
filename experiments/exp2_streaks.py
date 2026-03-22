"""
experiments/exp2_streaks.py
----------------------------
Experiment 2: Streak-Induced Behavior

Hypothesis:
    Participants exposed to controlled win streaks will subsequently
    over-select the risky option (hot-hand fallacy), while participants
    exposed to loss streaks will either under-select risky (excessive
    risk-aversion) or paradoxically over-select risky (loss-chasing /
    gambler's fallacy), depending on their psychological profile.

    In both cases, post-streak behavior deviates significantly from
    EV-maximizing strategy compared to the no-streak control.

Independent Variable: Streak condition (win streak / loss streak / no streak)
                       Streak length (3, 5, 7 trials)
Dependent Variable:   Deviation from EV-maximizing strategy in post-streak trials
Controlled:           Scenario set, expected values, participant profile

Measurement:
    - ev_deviation_post_streak:   mean EV left on table after streak
    - ev_deviation_control:       mean EV in no-streak condition
    - risky_rate_post_streak:     fraction risky choices after streak
    - risky_rate_control:         fraction risky choices with no streak
    - behavioral_shift:           risky_rate_post - risky_rate_control
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.scenario import Scenario, ScenarioLibrary, Frame
from core.engine import SimulationEngine, SimulationResult
from psychology.behavioral_model import BehavioralAgent, BehavioralProfile
from experiments.base_experiment import BaseExperiment, ExperimentConfig, ExperimentResult


class StreakExperiment(BaseExperiment):
    """
    Injects controlled win/loss streaks and measures post-streak behavior.

    Phase structure:
      [Baseline: 10 trials] → [Streak: N forced trials] → [Post-streak: 20 trials]

    The post-streak phase is compared against a no-streak control condition.
    """

    STREAK_LENGTHS = [3, 5, 7]
    CONDITIONS     = ["win_streak", "loss_streak", "no_streak"]

    def __init__(
        self,
        config:  ExperimentConfig,
        profile: BehavioralProfile = None,
    ):
        super().__init__(config)
        self.profile = profile or BehavioralProfile("default")

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------

    @property
    def scenarios(self) -> List[Scenario]:
        return ScenarioLibrary.standard_battery(Frame.NEUTRAL)

    @property
    def _base_scenario(self) -> Scenario:
        return ScenarioLibrary.classic_coin_flip()

    # ------------------------------------------------------------------
    # Run conditions
    # ------------------------------------------------------------------

    def run_condition(self, condition: str) -> SimulationResult:
        """Run a single agent through one condition."""
        agent = BehavioralAgent(profile=self.profile, seed=self.config.seed)

        if condition == "no_streak":
            return self.engine.run(
                scenarios = self.scenarios,
                strategy  = agent.as_strategy(),
                n_trials  = self.config.n_trials,
            )
        else:
            streak_type   = "win" if "win" in condition else "loss"
            streak_length = 5  # default
            _, post_result = self.engine.inject_streak(
                scenario    = self._base_scenario,
                streak_type = streak_type,
                length      = streak_length,
                strategy    = agent.as_strategy(),
            )
            return post_result

    def run_condition_population(
        self,
        condition:    str,
        streak_length: int,
        n_agents:     int,
        rng:          np.random.Generator,
    ) -> pd.DataFrame:
        """
        Run N agents through baseline → streak → post-streak.
        Returns post-streak trial data.
        """
        all_post: List[pd.DataFrame] = []
        scenarios = self.scenarios

        for agent_idx in range(n_agents):
            seed  = int(rng.integers(0, 999999))
            agent = BehavioralAgent(profile=self.profile, seed=seed)

            # Phase 1: Baseline
            baseline_result = self.engine.run(
                scenarios = scenarios,
                strategy  = agent.as_strategy(),
                n_trials  = 10,
            )
            baseline_history = baseline_result.trials

            if condition == "no_streak":
                # Control: just continue after baseline, no injected streak
                post_result = self.engine.run(
                    scenarios = scenarios,
                    strategy  = lambda s, h: agent.decide(s, baseline_history + h),
                    n_trials  = 20,
                )
                post_df = post_result.to_dataframe()
                post_df["baseline_pnl"] = baseline_result.total_pnl

            else:
                streak_type = "win" if "win" in condition else "loss"

                # Override agent's strategy to "see" baseline history
                strategy_with_history = lambda s, h: agent.decide(
                    s, baseline_history + h
                )

                streak_trials, post_result = self.engine.inject_streak(
                    scenario    = self._base_scenario,
                    streak_type = streak_type,
                    length      = streak_length,
                    strategy    = strategy_with_history,
                )

                post_df = post_result.to_dataframe()
                post_df["baseline_pnl"]   = baseline_result.total_pnl
                post_df["streak_type"]    = streak_type
                post_df["streak_length"]  = streak_length

            post_df["agent"]     = agent_idx
            post_df["condition"] = condition
            all_post.append(post_df)

        return pd.concat(all_post, ignore_index=True)

    # ------------------------------------------------------------------
    # Full experiment analysis
    # ------------------------------------------------------------------

    def analyze(self) -> ExperimentResult:
        rng      = np.random.default_rng(self.config.seed)
        n_agents = self.config.n_simulations

        all_dfs: List[pd.DataFrame] = []

        for streak_len in self.STREAK_LENGTHS:
            for condition in self.CONDITIONS:
                df = self.run_condition_population(condition, streak_len, n_agents, rng)
                df["streak_length"] = streak_len
                all_dfs.append(df)

        all_data = pd.concat(all_dfs, ignore_index=True)

        # --- Compute per-agent, per-condition summaries ---
        summary = (
            all_data.groupby(["condition", "streak_length", "agent"])
            .agg(
                risky_rate   = ("choice",    lambda x: (x == "risky").mean()),
                ev_deviation = ("ev_optimal", lambda x: (x - all_data.loc[x.index, "ev_chosen"]).mean()),
                total_pnl    = ("outcome",   "sum"),
                optimality   = ("is_optimal", "mean"),
            )
            .reset_index()
        )

        # --- Behavioral shift relative to no-streak control ---
        control_stats = (
            summary[summary["condition"] == "no_streak"]
            .groupby("streak_length")
            .agg(control_risky=("risky_rate", "mean"))
            .reset_index()
        )

        summary = summary.merge(control_stats, on="streak_length", how="left")
        summary["behavioral_shift"] = summary["risky_rate"] - summary["control_risky"]

        # --- Aggregate metrics ---
        win_df  = summary[summary["condition"] == "win_streak"]
        loss_df = summary[summary["condition"] == "loss_streak"]
        ctrl_df = summary[summary["condition"] == "no_streak"]

        metrics = {
            "risky_rate_control":            round(ctrl_df["risky_rate"].mean(), 4),
            "risky_rate_after_win_streak":   round(win_df["risky_rate"].mean(),  4),
            "risky_rate_after_loss_streak":  round(loss_df["risky_rate"].mean(), 4),
            "behavioral_shift_win":          round(win_df["behavioral_shift"].mean(),  4),
            "behavioral_shift_loss":         round(loss_df["behavioral_shift"].mean(), 4),
            "ev_deviation_control":          round(ctrl_df["ev_deviation"].mean(), 4),
            "ev_deviation_after_win":        round(win_df["ev_deviation"].mean(),  4),
            "ev_deviation_after_loss":       round(loss_df["ev_deviation"].mean(), 4),
            "n_agents":                      n_agents,
            "streak_lengths_tested":         str(self.STREAK_LENGTHS),
        }

        # Hypothesis: win streaks → positive behavioral shift
        #             loss streaks → non-zero behavioral shift
        win_shift  = win_df["behavioral_shift"].mean()
        loss_shift = loss_df["behavioral_shift"]
        supported  = (win_shift > 0.05) and (abs(loss_shift.mean()) > 0.05)

        return ExperimentResult(
            config               = self.config,
            raw_data             = all_data,
            aggregate_metrics    = metrics,
            hypothesis_supported = supported,
        )


# ------------------------------------------------------------------
# Default experiment factory
# ------------------------------------------------------------------

def make_streak_experiment(
    n_trials:      int = 40,
    n_simulations: int = 200,
    seed:          int = 42,
) -> StreakExperiment:
    config = ExperimentConfig(
        name           = "Experiment 2: Streak-Induced Behavior",
        hypothesis     = (
            "Post-streak behavior deviates significantly from EV-maximizing strategy. "
            "Win streaks induce hot-hand fallacy (increased risk-taking). "
            "Loss streaks induce either gambler's fallacy (increased risk-taking) "
            "or excessive risk-aversion, depending on agent profile."
        ),
        iv_description = "Streak condition (win streak / loss streak / no streak) and streak length (3, 5, 7)",
        dv_description = "Deviation from EV-maximizing strategy in post-streak trials",
        n_trials       = n_trials,
        n_simulations  = n_simulations,
        seed           = seed,
        notes          = "Tests hot-hand fallacy, gambler's fallacy, and loss-chasing",
    )
    return StreakExperiment(config=config)
