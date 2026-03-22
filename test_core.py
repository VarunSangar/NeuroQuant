"""
tests/test_core.py
-------------------
Unit tests for NeuroQuant core modules.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from core.scenario import Scenario, Outcome, Frame, ScenarioLibrary
from core.engine import SimulationEngine
from core.strategy import rational_strategy, make_risk_averse, risk_seeking_strategy
from psychology.prospect_theory import ProspectValueFunction, ProbabilityWeighting, ProspectCalculator
from psychology.biases import DecisionFatigue, RecencyBias, StreakSensitivity
from psychology.behavioral_model import BehavioralAgent, BehavioralProfile
from agents.adaptive_agent import AdaptiveAgent
from analysis.metrics import compute_full_metrics


# ------------------------------------------------------------------
# Scenario
# ------------------------------------------------------------------

class TestScenario:

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError):
            Scenario(
                description="bad",
                risky_outcomes=[Outcome(100, 0.4), Outcome(-10, 0.4)],   # sums to 0.8
                safe_outcomes=[Outcome(10, 1.0)],
            )

    def test_ev_calculation(self):
        s = Scenario(
            description="test",
            risky_outcomes=[Outcome(100, 0.5), Outcome(-20, 0.5)],
            safe_outcomes=[Outcome(30, 1.0)],
        )
        assert abs(s.ev("risky") - 40.0) < 1e-6
        assert abs(s.ev("safe")  - 30.0) < 1e-6

    def test_optimal_choice(self):
        s = Scenario(
            description="test",
            risky_outcomes=[Outcome(100, 0.5), Outcome(-20, 0.5)],
            safe_outcomes=[Outcome(30, 1.0)],
        )
        assert s.optimal_choice() == "risky"   # EV(risky)=40 > EV(safe)=30

    def test_variance_nonneg(self):
        s = ScenarioLibrary.classic_coin_flip()
        assert s.variance("risky") >= 0
        assert s.variance("safe")  >= 0

    def test_safe_option_variance(self):
        s = ScenarioLibrary.classic_coin_flip()
        assert s.variance("safe") == pytest.approx(0.0)

    def test_sample_returns_valid_value(self):
        s = ScenarioLibrary.classic_coin_flip()
        rng  = np.random.default_rng(0)
        vals = {o.value for o in s.risky_outcomes}
        for _ in range(50):
            assert s.sample("risky", rng) in vals

    def test_framing_pair_same_ev(self):
        gain_s, loss_s = ScenarioLibrary.framing_pair()
        assert abs(gain_s.ev("risky") - loss_s.ev("risky")) < 1e-6
        assert abs(gain_s.ev("safe")  - loss_s.ev("safe"))  < 1e-6

    def test_standard_battery_length(self):
        assert len(ScenarioLibrary.standard_battery()) == 5


# ------------------------------------------------------------------
# Simulation Engine
# ------------------------------------------------------------------

class TestEngine:

    def test_run_returns_correct_trial_count(self):
        engine    = SimulationEngine(seed=0)
        scenarios = ScenarioLibrary.standard_battery()
        result    = engine.run(scenarios, rational_strategy, n_trials=25)
        assert len(result.trials) == 25

    def test_optimality_rate_rational(self):
        """Rational strategy should always be optimal."""
        engine    = SimulationEngine(seed=0)
        scenarios = ScenarioLibrary.standard_battery()
        result    = engine.run(scenarios, rational_strategy, n_trials=100)
        assert result.optimality_rate == pytest.approx(1.0)

    def test_ev_deviation_rational(self):
        """Rational agent should have zero EV deviation."""
        engine    = SimulationEngine(seed=0)
        scenarios = ScenarioLibrary.standard_battery()
        result    = engine.run(scenarios, rational_strategy, n_trials=100)
        assert result.ev_deviation == pytest.approx(0.0)

    def test_monte_carlo_shape(self):
        engine    = SimulationEngine(seed=0)
        scenarios = ScenarioLibrary.standard_battery()
        df        = engine.monte_carlo(scenarios, rational_strategy, n_runs=20, n_trials=10)
        assert len(df) == 20
        assert "total_pnl" in df.columns

    def test_streak_injection_returns_history(self):
        engine   = SimulationEngine(seed=0)
        scenario = ScenarioLibrary.classic_coin_flip()
        streak_h, post = engine.inject_streak(scenario, "win", 5, rational_strategy)
        assert len(streak_h) == 5
        for t in streak_h:
            assert t.outcome > 0   # All outcomes should be forced wins


# ------------------------------------------------------------------
# Prospect Theory
# ------------------------------------------------------------------

class TestProspectTheory:

    def test_value_function_gains(self):
        vf = ProspectValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)
        assert vf(100) > 0
        assert vf(0)   == 0.0

    def test_value_function_losses(self):
        vf = ProspectValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)
        assert vf(-100) < 0

    def test_loss_aversion(self):
        vf = ProspectValueFunction(lambda_=2.25)
        # Pain from -$100 should be ~2.25x joy from +$100
        assert abs(vf(-100)) > abs(vf(100))

    def test_probability_weighting_boundary(self):
        pw = ProbabilityWeighting()
        assert pw.weight_gain(0) == pytest.approx(0.0)
        assert pw.weight_gain(1) == pytest.approx(1.0)

    def test_probability_weighting_inverse_s(self):
        """Small probs overweighted, large probs underweighted."""
        pw = ProbabilityWeighting(gamma=0.61)
        assert pw.weight_gain(0.01) > 0.01  # Overweighted
        assert pw.weight_gain(0.9)  < 0.9   # Underweighted

    def test_prospect_calculator_choice(self):
        calc     = ProspectCalculator()
        scenario = ScenarioLibrary.classic_coin_flip()
        choice   = calc.subjective_choice(scenario)
        assert choice in ("risky", "safe")


# ------------------------------------------------------------------
# Psychological Biases
# ------------------------------------------------------------------

class TestBiases:

    def test_fatigue_decay(self):
        f = DecisionFatigue(decay_rate=0.1, floor=0.4)
        assert f.rationality(0)   == pytest.approx(1.0)
        assert f.rationality(100) == pytest.approx(0.4, abs=0.05)
        assert f.rationality(50)  < f.rationality(0)

    def test_fatigue_floor_respected(self):
        f = DecisionFatigue(decay_rate=0.5, floor=0.3)
        assert f.rationality(1000) >= 0.3

    def test_recency_bias_no_history(self):
        r = RecencyBias()
        assert r.perceived_win_prob([], 0.5) == pytest.approx(0.5)

    def test_streak_sensitivity_no_streak(self):
        ss = StreakSensitivity(streak_threshold=3)
        assert ss.risk_adjustment(0) == 0.0
        assert ss.risk_adjustment(2) == 0.0

    def test_streak_sensitivity_above_threshold(self):
        ss = StreakSensitivity(streak_threshold=3, hot_hand_weight=1.0, gamblers_weight=0.0)
        adj = ss.risk_adjustment(5)
        assert adj > 0   # Pure hot-hand: win streak → more risk


# ------------------------------------------------------------------
# Behavioral Agent
# ------------------------------------------------------------------

class TestBehavioralAgent:

    def test_agent_returns_valid_choice(self):
        agent    = BehavioralAgent(seed=0)
        scenario = ScenarioLibrary.classic_coin_flip()
        choice   = agent.decide(scenario, [])
        assert choice in ("risky", "safe")

    def test_rational_profile_is_optimal(self):
        """With rationality=1.0, agent should always choose optimally."""
        profile  = BehavioralProfile.rational_baseline()
        agent    = BehavioralAgent(profile=profile, seed=0)
        scenario = ScenarioLibrary.classic_coin_flip()

        choices  = [agent.decide(scenario, []) for _ in range(50)]
        opt      = scenario.optimal_choice()
        # All choices should match optimal
        assert all(c == opt for c in choices)

    def test_rationality_trajectory_decays(self):
        profile    = BehavioralProfile.fatigued_analyst()
        agent      = BehavioralAgent(profile=profile, seed=0)
        trajectory = agent.rationality_trajectory(50)
        assert trajectory[0] > trajectory[-1]  # Must decay


# ------------------------------------------------------------------
# Adaptive Agent
# ------------------------------------------------------------------

class TestAdaptiveAgent:

    def test_agent_runs_episode(self):
        ai        = AdaptiveAgent(seed=0)
        engine    = SimulationEngine(seed=0)
        scenarios = ScenarioLibrary.standard_battery()
        log       = ai.run_episode(scenarios, n_trials=20, engine=engine)
        assert len(log.choices)  == 20
        assert len(log.rewards)  == 20
        assert len(log.epsilons) == 20

    def test_epsilon_decreases_over_time(self):
        ai = AdaptiveAgent(epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.1, seed=0)
        e0 = ai._current_epsilon()
        for _ in range(100):
            ai._step += 1
        e1 = ai._current_epsilon()
        assert e1 < e0

    def test_reset_clears_state(self):
        ai = AdaptiveAgent(seed=0)
        ai._step = 100
        ai.reset()
        assert ai.step_count == 0


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

class TestMetrics:

    def setup_method(self):
        """Generate a small dataset for metric testing."""
        engine    = SimulationEngine(seed=0)
        scenarios = ScenarioLibrary.standard_battery()
        result    = engine.run(scenarios, rational_strategy, n_trials=30)
        self.df   = result.to_dataframe()

    def test_compute_full_metrics_keys(self):
        m = compute_full_metrics(self.df)
        assert "optimality_rate"  in m
        assert "ev_deviation"     in m
        assert "sharpe_ratio"     in m
        assert "max_drawdown"     in m

    def test_rational_agent_optimality(self):
        m = compute_full_metrics(self.df)
        assert m["optimality_rate"] == pytest.approx(1.0)

    def test_ev_deviation_nonneg_rational(self):
        m = compute_full_metrics(self.df)
        assert m["ev_deviation"] == pytest.approx(0.0, abs=1e-6)
