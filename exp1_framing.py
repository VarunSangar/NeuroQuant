"""
experiments/exp1_framing.py
----------------------------
Experiment 1: Gain vs Loss Framing

Hypothesis:
    Participants exhibit greater risk-AVERSION when choices are framed as
    gains and greater risk-SEEKING when framed as losses, even when the
    underlying probability distributions and expected values are identical.
    (Replication of Kahneman & Tversky's Asian Disease Problem paradigm.)

Independent Variable: Framing (Gain Frame vs Loss Frame)
Dependent Variable:   Proportion of risky choices
Controlled:           Probability distributions, expected values, scenario order,
                      number of trials, participant profile

Measurement:
    - risky_rate_gain:  fraction choosing risky under gain frame
    - risky_rate_loss:  fraction choosing risky under loss frame
    - framing_effect:   risky_rate_loss - risky_rate_gain (should be positive)
    - ev_deviation:     EV foregone under each frame
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.scenario import Scenario, ScenarioLibrary, Frame
from core.engine import SimulationEngine, SimulationResult
from core.strategy import rational_strategy
from psychology.behavioral_model import BehavioralAgent, BehavioralProfile
from experiments.base_experiment import BaseExperiment, ExperimentConfig, ExperimentResult


class FramingExperiment(BaseExperiment):
    """
    Presents identical scenarios under gain and loss framing.
    Runs N simulated agents through each condition and compares
    risk-taking rates and EV deviations across frames.
    """

    CONDITION_GAIN = "gain"
    CONDITION_LOSS = "loss"

    def __init__(
        self,
        config: ExperimentConfig,
        profile: BehavioralProfile = None,
    ):
        super().__init__(config)
        self.profile = profile or BehavioralProfile.loss_averse_investor()

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------

    @property
    def scenarios(self) -> List[Scenario]:
        return ScenarioLibrary.standard_battery(Frame.NEUTRAL)

    def _scenarios_for_condition(self, condition: str) -> List[Scenario]:
        frame = Frame.GAIN if condition == self.CONDITION_GAIN else Frame.LOSS
        return ScenarioLibrary.standard_battery(frame)

    # ------------------------------------------------------------------
    # Run one condition
    # ------------------------------------------------------------------

    def run_condition(self, condition: str) -> SimulationResult:
        scenarios = self._scenarios_for_condition(condition)
        agent     = BehavioralAgent(profile=self.profile, seed=self.config.seed)
        return self.engine.run(
            scenarios = scenarios,
            strategy  = agent.as_strategy(),
            n_trials  = self.config.n_trials,
        )

    def run_condition_population(
        self,
        condition:  str,
        n_agents:   int,
        rng:        np.random.Generator,
    ) -> pd.DataFrame:
        """Run N independent agents through one condition. Returns combined DataFrame."""
        scenarios = self._scenarios_for_condition(condition)
        all_dfs   = []

        for agent_idx in range(n_agents):
            agent  = BehavioralAgent(
                profile = self.profile,
                seed    = int(rng.integers(0, 999999)),
            )
            result = self.engine.run(
                scenarios = scenarios,
                strategy  = agent.as_strategy(),
                n_trials  = self.config.n_trials,
            )
            df          = result.to_dataframe()
            df["agent"] = agent_idx
            df["frame"] = condition
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Full experiment analysis
    # ------------------------------------------------------------------

    def analyze(self) -> ExperimentResult:
        rng      = np.random.default_rng(self.config.seed)
        n_agents = self.config.n_simulations

        # Run both conditions
        df_gain = self.run_condition_population(self.CONDITION_GAIN, n_agents, rng)
        df_loss = self.run_condition_population(self.CONDITION_LOSS, n_agents, rng)

        all_data = pd.concat([df_gain, df_loss], ignore_index=True)

        # --- Per-agent summary statistics ---
        def agent_summary(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df.groupby("agent")
                .agg(
                    risky_rate   = ("choice",     lambda x: (x == "risky").mean()),
                    ev_deviation = ("ev_optimal",  lambda x: (x - df.loc[x.index, "ev_chosen"]).mean()),
                    optimality   = ("is_optimal", "mean"),
                    total_pnl    = ("outcome",    "sum"),
                )
                .reset_index()
            )

        summary_gain = agent_summary(df_gain)
        summary_loss = agent_summary(df_loss)

        # --- Key metrics ---
        risky_gain  = float(summary_gain["risky_rate"].mean())
        risky_loss  = float(summary_loss["risky_rate"].mean())
        framing_effect = risky_loss - risky_gain   # > 0 supports hypothesis

        # T-test equivalent: check if the means differ substantially
        gain_arr = summary_gain["risky_rate"].values
        loss_arr = summary_loss["risky_rate"].values

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((gain_arr.std()**2 + loss_arr.std()**2) / 2)
        cohens_d   = (loss_arr.mean() - gain_arr.mean()) / pooled_std if pooled_std > 0 else 0.0

        metrics = {
            "risky_rate_gain_frame":    round(risky_gain, 4),
            "risky_rate_loss_frame":    round(risky_loss, 4),
            "framing_effect":           round(framing_effect, 4),
            "cohens_d":                 round(cohens_d, 4),
            "ev_deviation_gain_frame":  round(summary_gain["ev_deviation"].mean(), 4),
            "ev_deviation_loss_frame":  round(summary_loss["ev_deviation"].mean(), 4),
            "optimality_rate_gain":     round(summary_gain["optimality"].mean(), 4),
            "optimality_rate_loss":     round(summary_loss["optimality"].mean(), 4),
            "pnl_gain_frame":           round(summary_gain["total_pnl"].mean(), 2),
            "pnl_loss_frame":           round(summary_loss["total_pnl"].mean(), 2),
            "n_agents":                 n_agents,
            "n_trials_per_agent":       self.config.n_trials,
        }

        # Hypothesis: framing effect > 0 and Cohen's d > 0.2 (small effect)
        supported = (framing_effect > 0.05) and (cohens_d > 0.2)

        return ExperimentResult(
            config               = self.config,
            raw_data             = all_data,
            aggregate_metrics    = metrics,
            hypothesis_supported = supported,
        )


# ------------------------------------------------------------------
# Default experiment factory
# ------------------------------------------------------------------

def make_framing_experiment(
    n_trials:      int = 40,
    n_simulations: int = 200,
    seed:          int = 42,
) -> FramingExperiment:
    config = ExperimentConfig(
        name           = "Experiment 1: Gain vs Loss Framing",
        hypothesis     = (
            "Participants choose the risky option more frequently under loss framing "
            "than under gain framing, despite identical underlying probability distributions."
        ),
        iv_description = "Frame condition: Gain Frame vs Loss Frame",
        dv_description = "Proportion of risky choices per participant",
        n_trials       = n_trials,
        n_simulations  = n_simulations,
        seed           = seed,
        notes          = "Kahneman-Tversky framing paradigm applied to financial scenarios",
    )
    return FramingExperiment(config=config)
