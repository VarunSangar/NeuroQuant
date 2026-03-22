"""
sample_data/generate_sample.py
--------------------------------
Generates a realistic synthetic dataset of human-like decisions
across multiple behavioral profiles. Produces:

  sample_data/
    decisions_all.csv        — Combined multi-participant dataset
    decisions_human.csv      — Human-profile agent decisions
    decisions_rational.csv   — Rational agent decisions
    decisions_ai.csv         — Adaptive AI agent decisions
    summary_stats.json       — Aggregate statistics per agent type
"""

import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from core.scenario import ScenarioLibrary, Frame
from core.engine import SimulationEngine
from core.strategy import rational_strategy, make_risk_averse
from psychology.behavioral_model import BehavioralAgent, BehavioralProfile
from agents.adaptive_agent import AdaptiveAgent
from analysis.metrics import compute_full_metrics


OUTPUT_DIR  = os.path.join(os.path.dirname(__file__))
N_AGENTS    = 30     # Participants per profile
N_TRIALS    = 60     # Trials per participant
SEED        = 2024


def generate_agent_dataset(
    profile_name: str,
    profile:      BehavioralProfile,
    n_agents:     int,
    n_trials:     int,
    engine:       SimulationEngine,
    scenarios,
    rng:          np.random.Generator,
) -> pd.DataFrame:
    """Generate decisions from N agents sharing a behavioral profile."""
    all_dfs = []
    for i in range(n_agents):
        agent  = BehavioralAgent(profile=profile, seed=int(rng.integers(0, 999999)))
        result = engine.run(
            scenarios = scenarios,
            strategy  = agent.as_strategy(),
            n_trials  = n_trials,
        )
        df                = result.to_dataframe()
        df["agent_id"]    = f"{profile_name}_{i:03d}"
        df["agent_type"]  = profile_name
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def generate_rational_dataset(n_agents, n_trials, engine, scenarios, rng):
    """Rational baseline agents."""
    all_dfs = []
    for i in range(n_agents):
        result = engine.run(
            scenarios = scenarios,
            strategy  = rational_strategy,
            n_trials  = n_trials,
        )
        df               = result.to_dataframe()
        df["agent_id"]   = f"rational_{i:03d}"
        df["agent_type"] = "rational"
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def generate_ai_dataset(n_agents, n_trials, engine, scenarios, rng):
    """Adaptive AI agent dataset — runs with online learning."""
    all_dfs = []
    for i in range(n_agents):
        ai   = AdaptiveAgent(seed=int(rng.integers(0, 999999)))
        log  = ai.run_episode(
            scenarios = scenarios,
            n_trials  = n_trials,
            engine    = engine,
            rng       = np.random.default_rng(int(rng.integers(0, 999999))),
        )
        df = log.to_dataframe()
        # Enrich with EV columns by re-running scenario sequence
        ev_chosen   = []
        ev_optimal  = []
        is_optimal  = []
        for j, choice in enumerate(log.choices):
            s = scenarios[j % len(scenarios)]
            ev_chosen.append(s.ev(choice))
            opt = s.optimal_choice()
            ev_optimal.append(s.ev(opt))
            is_optimal.append(choice == opt)

        df["ev_chosen"]   = ev_chosen
        df["ev_optimal"]  = ev_optimal
        df["is_optimal"]  = is_optimal
        df["agent_id"]    = f"adaptive_ai_{i:03d}"
        df["agent_type"]  = "adaptive_ai"
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng       = np.random.default_rng(SEED)
    engine    = SimulationEngine(seed=SEED)
    scenarios = ScenarioLibrary.standard_battery(Frame.NEUTRAL)

    print("Generating sample data...")
    print(f"  Profiles: loss_averse, overconfident, fatigued, rational, adaptive_ai")
    print(f"  N agents: {N_AGENTS} per profile | N trials: {N_TRIALS}")
    print()

    profiles = {
        "loss_averse":     BehavioralProfile.loss_averse_investor(),
        "overconfident":   BehavioralProfile.overconfident_trader(),
        "fatigued":        BehavioralProfile.fatigued_analyst(),
    }

    all_dfs = []

    # Human profiles
    for name, profile in profiles.items():
        print(f"  → Generating {name}...")
        df = generate_agent_dataset(name, profile, N_AGENTS, N_TRIALS, engine, scenarios, rng)
        all_dfs.append(df)
        df.to_csv(os.path.join(OUTPUT_DIR, f"decisions_{name}.csv"), index=False)

    # Rational baseline
    print("  → Generating rational baseline...")
    df_rational = generate_rational_dataset(N_AGENTS, N_TRIALS, engine, scenarios, rng)
    all_dfs.append(df_rational)
    df_rational.to_csv(os.path.join(OUTPUT_DIR, "decisions_rational.csv"), index=False)

    # Adaptive AI
    print("  → Generating adaptive AI...")
    df_ai = generate_ai_dataset(N_AGENTS, N_TRIALS, engine, scenarios, rng)
    all_dfs.append(df_ai)
    df_ai.to_csv(os.path.join(OUTPUT_DIR, "decisions_ai.csv"), index=False)

    # Combined
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "decisions_all.csv"), index=False)

    # Summary statistics per agent type
    summary = {}
    for agent_type in df_all["agent_type"].unique():
        subset = df_all[df_all["agent_type"] == agent_type]
        metrics = compute_full_metrics(subset)
        summary[agent_type] = metrics

    with open(os.path.join(OUTPUT_DIR, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("  Done! Files written to sample_data/")
    print()

    # Print comparison table
    print(f"  {'Agent Type':<20} {'Optimality':>12} {'EV Deviation':>14} {'Sharpe':>8}")
    print(f"  {'-'*60}")
    for agent_type, m in summary.items():
        print(f"  {agent_type:<20} {m['optimality_rate']:>12.1%} {m['ev_deviation']:>14.4f} {m['sharpe_ratio']:>8.3f}")


if __name__ == "__main__":
    main()
