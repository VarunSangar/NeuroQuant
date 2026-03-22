"""
ui/app.py
----------
NeuroQuant Interactive UI — built with Streamlit.

Presents trading scenarios to a live user, tracks decisions in real-time,
and renders live analytics after each choice. Supports both human play
and automated agent simulation for comparison.

Run with:
    streamlit run ui/app.py
"""
import sys
import os

# Add the directory containing app.py to the path
# On Streamlit Cloud this resolves to /mount/src/neuroquant/
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.scenario import ScenarioLibrary, Frame, Scenario
from core.engine import SimulationEngine
from core.strategy import rational_strategy, make_risk_averse
from psychology.behavioral_model import BehavioralAgent, BehavioralProfile
from agents.adaptive_agent import AdaptiveAgent
from data.session_store import Session
from analysis.metrics import compute_full_metrics
from visualization.plots import (
    plot_equity_curve,
    plot_risk_taking_trajectory,
    plot_ev_deviation,
    plot_session_dashboard,
    plot_prospect_theory_curves,
)


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title  = "NeuroQuant",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #457B9D;
    }
    .scenario-box {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .risky-choice { border-left: 5px solid #E63946; }
    .safe-choice  { border-left: 5px solid #457B9D; }
    .outcome-win  { color: #2A9D8F; font-weight: bold; font-size: 1.2rem; }
    .outcome-loss { color: #E63946; font-weight: bold; font-size: 1.2rem; }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------

def init_state():
    defaults = {
        "page":            "home",
        "session":         None,
        "scenarios":       [],
        "trial_index":     0,
        "history":         [],
        "session_started": False,
        "session_done":    False,
        "last_outcome":    None,
        "decision_start":  None,
        "frame":           "neutral",
        "n_trials":        20,
        "agent_mode":      "human",
        "sim_agent":       None,
        "show_ev":         False,
        "rng":             np.random.default_rng(int(time.time())),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def reset_session():
    keys = ["session", "scenarios", "trial_index", "history",
            "session_started", "session_done", "last_outcome", "decision_start"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    init_state()


def get_scenarios(frame_str: str, n: int) -> list:
    frame_map = {"neutral": Frame.NEUTRAL, "gain": Frame.GAIN, "loss": Frame.LOSS}
    frame     = frame_map.get(frame_str, Frame.NEUTRAL)
    battery   = ScenarioLibrary.standard_battery(frame)
    rng       = np.random.default_rng(42)
    idx       = [i % len(battery) for i in range(n)]
    rng.shuffle(idx)
    return [battery[i] for i in idx]


def record_choice(scenario: Scenario, choice: str) -> dict:
    """Process a choice and return result dict."""
    rng     = st.session_state["rng"]
    outcome = scenario.sample(choice, rng=rng)
    ev_c    = scenario.ev(choice)
    opt     = scenario.optimal_choice()
    ev_o    = scenario.ev(opt)
    is_opt  = choice == opt

    # Compute streak
    history = st.session_state["history"]
    streak  = history[-1]["streak"] if history else 0
    if outcome > 0:
        streak = max(streak + 1, 1)
    elif outcome < 0:
        streak = min(streak - 1, -1)
    else:
        streak = 0

    record = {
        "trial":       st.session_state["trial_index"],
        "scenario_id": scenario.scenario_id,
        "frame":       scenario.frame.value,
        "choice":      choice,
        "outcome":     outcome,
        "ev_chosen":   ev_c,
        "ev_optimal":  ev_o,
        "is_optimal":  is_opt,
        "streak":      streak,
        "pnl":         (history[-1]["pnl"] if history else 0) + outcome,
    }

    st.session_state["history"].append(record)
    st.session_state["trial_index"] += 1
    st.session_state["last_outcome"] = record

    if st.session_state["trial_index"] >= st.session_state["n_trials"]:
        st.session_state["session_done"] = True

    return record


def history_to_df() -> pd.DataFrame:
    h = st.session_state["history"]
    if not h:
        return pd.DataFrame()
    return pd.DataFrame(h).rename(columns={
        "streak": "current_streak",
        "pnl":    "cumulative_pnl",
    })


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def page_home():
    st.markdown('<p class="main-header">🧠 NeuroQuant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Modeling Human Decision-Making Under Uncertainty in Simulated Trading Environments</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### About")
        st.write(
            "NeuroQuant places you in repeated trading scenarios and measures how "
            "your decisions compare to mathematically optimal choices. "
            "The system tracks behavioral patterns including loss aversion, "
            "recency bias, and decision fatigue."
        )

        st.markdown("### Configure Your Session")

        n_trials = st.slider("Number of trials", 10, 50, 20, step=5)
        frame    = st.selectbox(
            "Framing condition",
            options=["neutral", "gain", "loss"],
            format_func=lambda x: {
                "neutral": "Neutral (standard probabilities)",
                "gain":    "Gain Frame (outcomes presented as potential gains)",
                "loss":    "Loss Frame (outcomes presented as potential losses)",
            }[x]
        )
        show_ev = st.checkbox("Show expected value hints (reduces realism)", value=False)

        st.markdown("#### Optional: Run simulation agent alongside you")
        agent_mode = st.selectbox(
            "Comparison agent",
            ["none", "rational", "behavioral", "adaptive_ai"],
            format_func=lambda x: {
                "none":        "None (human only)",
                "rational":    "Rational (EV-maximizing)",
                "behavioral":  "Behavioral (loss-averse profile)",
                "adaptive_ai": "Adaptive AI (learning agent)",
            }[x]
        )

        if st.button("▶  Start Session", type="primary"):
            reset_session()
            st.session_state["n_trials"]   = n_trials
            st.session_state["frame"]      = frame
            st.session_state["show_ev"]    = show_ev
            st.session_state["agent_mode"] = agent_mode
            st.session_state["scenarios"]  = get_scenarios(frame, n_trials)
            st.session_state["session_started"] = True
            st.session_state["page"] = "experiment"
            st.rerun()

    with col2:
        st.markdown("### What We Measure")
        st.info("📊 **Optimality Rate** — How often you choose the EV-maximizing option")
        st.info("💸 **EV Deviation** — Value left on the table per trial")
        st.info("📈 **Risk Trajectory** — How your risk appetite changes over time")
        st.info("🔁 **Streak Effects** — How win/loss streaks alter your choices")

        st.markdown("### The Science")
        st.write(
            "Based on Kahneman-Tversky Prospect Theory (1979). "
            "People overweight losses relative to equivalent gains (loss aversion), "
            "overweight small probabilities, and update beliefs based on recent "
            "outcomes rather than true probabilities."
        )


def page_experiment():
    scenarios = st.session_state["scenarios"]
    idx       = st.session_state["trial_index"]
    n         = st.session_state["n_trials"]

    if st.session_state["session_done"]:
        st.session_state["page"] = "results"
        st.rerun()

    # Header
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.markdown(f"### Trial {idx + 1} of {n}")
        st.progress((idx) / n)
    with col_h2:
        history = st.session_state["history"]
        pnl     = history[-1]["pnl"] if history else 0
        color   = "green" if pnl >= 0 else "red"
        st.metric("Current P&L", f"${pnl:+.2f}")
    with col_h3:
        opt_rate = sum(1 for h in history if h["is_optimal"]) / len(history) if history else 0
        st.metric("Optimality", f"{opt_rate:.0%}")

    # Show last outcome if available
    last = st.session_state["last_outcome"]
    if last:
        if last["outcome"] > 0:
            st.success(f"✓  Previous outcome: **+${last['outcome']:.2f}** "
                       f"({'✓ Optimal' if last['is_optimal'] else '✗ Suboptimal'})")
        elif last["outcome"] < 0:
            st.error(f"✗  Previous outcome: **${last['outcome']:.2f}** "
                     f"({'✓ Optimal' if last['is_optimal'] else '✗ Suboptimal'})")
        else:
            st.info("Previous outcome: $0.00")

    st.markdown("---")

    # Current scenario
    scenario   = scenarios[idx]
    show_ev    = st.session_state["show_ev"]

    st.markdown("### 📋 Current Decision")

    # Scenario description
    with st.container():
        st.markdown(f'<div class="scenario-box">', unsafe_allow_html=True)
        st.write(scenario.description)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎲 Option A — Risky")
            for o in scenario.risky_outcomes:
                sign = "+" if o.value >= 0 else ""
                st.write(f"• **{o.probability:.0%}** chance of **${sign}{o.value:.0f}**")
            if show_ev:
                st.caption(f"Expected value: ${scenario.ev('risky'):+.2f}")

        with c2:
            st.markdown("#### 🛡️ Option B — Safe")
            for o in scenario.safe_outcomes:
                sign = "+" if o.value >= 0 else ""
                st.write(f"• **{o.probability:.0%}** chance of **${sign}{o.value:.0f}**")
            if show_ev:
                st.caption(f"Expected value: ${scenario.ev('safe'):+.2f}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Choice buttons
    st.markdown("### Make Your Choice")
    btn1, btn2, btn_skip = st.columns([2, 2, 1])

    with btn1:
        if st.button("🎲  Choose Risky (Option A)", type="primary"):
            record_choice(scenario, "risky")
            st.rerun()

    with btn2:
        if st.button("🛡️  Choose Safe (Option B)"):
            record_choice(scenario, "safe")
            st.rerun()

    with btn_skip:
        if st.button("⏩ Skip"):
            record_choice(scenario, scenario.optimal_choice())  # Count as optimal
            st.rerun()

    # Mini live chart (after a few trials)
    if len(history) >= 5:
        st.markdown("---")
        st.markdown("#### Live Performance")
        df  = history_to_df()
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(df["cumulative_pnl"].values, color="#457B9D", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.fill_between(range(len(df)), df["cumulative_pnl"].values, alpha=0.1, color="#457B9D")
        ax.set_xlabel("Trial"); ax.set_ylabel("P&L ($)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Sidebar: running stats
    with st.sidebar:
        st.markdown("### 📊 Live Stats")
        if history:
            df = history_to_df()
            metrics = compute_full_metrics(df)
            st.metric("Trials Completed",  idx)
            st.metric("Total P&L",         f"${metrics['total_pnl']:+.2f}")
            st.metric("Optimality Rate",   f"{metrics['optimality_rate']:.0%}")
            st.metric("Avg EV Deviation",  f"${metrics['ev_deviation']:+.2f}")
            st.metric("Win Rate",          f"{metrics['win_rate']:.0%}")
            streak = history[-1]["streak"]
            if streak > 0:
                st.success(f"Win streak: {streak} 🔥")
            elif streak < 0:
                st.error(f"Loss streak: {abs(streak)} 🔻")

        if st.button("↩  Quit & See Results"):
            st.session_state["session_done"] = True
            st.rerun()


def page_results():
    history = st.session_state["history"]
    if not history:
        st.warning("No decisions recorded.")
        if st.button("Start Over"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    df = history_to_df()
    st.markdown('<p class="main-header">📊 Session Results</p>', unsafe_allow_html=True)

    # Top metrics
    metrics = compute_full_metrics(df)
    cols    = st.columns(5)
    kpis    = [
        ("Total P&L",       f"${metrics['total_pnl']:+.2f}"),
        ("Optimality",      f"{metrics['optimality_rate']:.0%}"),
        ("Risky Rate",      f"{metrics['risky_choice_rate']:.0%}"),
        ("EV Deviation",    f"${metrics['ev_deviation']:+.4f}"),
        ("Sharpe",          f"{metrics['sharpe_ratio']:.3f}"),
    ]
    for col, (label, val) in zip(cols, kpis):
        col.metric(label, val)

    st.markdown("---")

    # Dashboard
    st.markdown("### Full Session Dashboard")
    fig = plot_session_dashboard(df, title="Your Session")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Trajectory
    st.markdown("### Risk-Taking Trajectory")
    fig2 = plot_risk_taking_trajectory(df, title="Your Risk-Taking Over Time")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Interpretation
    st.markdown("### Behavioral Interpretation")
    ic1, ic2, ic3 = st.columns(3)

    with ic1:
        rd = metrics.get("rationality_decay_slope", 0)
        if rd < -0.002:
            st.warning(f"**Decision Fatigue Detected**\nYour optimality declined over time (slope={rd:.4f}). "
                       "This is consistent with cognitive depletion.")
        else:
            st.success("**No significant decision fatigue** detected.")

    with ic2:
        rbi = metrics.get("recency_bias_index", 0)
        if rbi > 0.6:
            st.warning(f"**Recency Bias Present** (index={rbi:.2f})\n"
                       "Your choices were predicted by recent outcomes more than chance.")
        else:
            st.success("**Low recency bias** detected.")

    with ic3:
        src = metrics.get("streak_risk_correlation", 0)
        if src > 0.1:
            st.info(f"**Hot-Hand Tendency** (r={src:.2f})\n"
                    "You tended to take more risk after winning streaks.")
        elif src < -0.1:
            st.info(f"**Gambler's Fallacy Tendency** (r={src:.2f})\n"
                    "You tended to change behavior after streaks.")
        else:
            st.success("**Streak-neutral behavior** — streaks did not strongly predict your choices.")

    # Comparison with rational agent
    st.markdown("### Comparison: You vs Rational Agent")

    engine    = SimulationEngine(seed=999)
    scenarios = st.session_state["scenarios"]
    rat_result = engine.run(scenarios, rational_strategy, n_trials=len(history))
    rat_df     = rat_result.to_dataframe()

    dfs_compare = {"You": df, "Rational Agent": rat_df}
    fig3 = plot_equity_curve(dfs_compare, title="Equity: You vs Rational Agent")
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Decision log
    st.markdown("### Full Decision Log")
    display_cols = ["trial", "choice", "outcome", "ev_chosen", "ev_optimal",
                    "is_optimal", "cumulative_pnl", "current_streak"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].style
          .applymap(lambda v: "color: green" if v == True else ("color: red" if v == False else ""),
                    subset=["is_optimal"] if "is_optimal" in available else [])
          .format({c: "{:.2f}" for c in ["outcome", "ev_chosen", "ev_optimal", "cumulative_pnl"]
                   if c in available}),
        use_container_width=True,
    )

    # Download
    csv = df.to_csv(index=False)
    st.download_button("⬇ Download Decision Log (CSV)", csv, "neuroquant_decisions.csv", "text/csv")

    st.markdown("---")
    if st.button("🔄  Start New Session", type="primary"):
        reset_session()
        st.session_state["page"] = "home"
        st.rerun()


def page_theory():
    st.markdown("### 📚 Prospect Theory Reference")
    st.write(
        "These plots illustrate the core psychological models underlying NeuroQuant's "
        "behavioral simulation. The value function captures diminishing sensitivity and "
        "loss aversion; the probability weighting function captures distortions in how "
        "people perceive probabilities."
    )
    fig = plot_prospect_theory_curves()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    **Key Parameters (Tversky & Kahneman, 1992):**
    | Parameter | Value | Interpretation |
    |-----------|-------|----------------|
    | α (gain sensitivity) | 0.88 | Diminishing marginal joy from gains |
    | β (loss sensitivity) | 0.88 | Diminishing marginal pain from losses |
    | λ (loss aversion)    | 2.25 | Losses hurt ~2.25× more than equivalent gains |
    | γ (gain weighting)   | 0.61 | Overweighting of small probabilities in gain domain |
    | δ (loss weighting)   | 0.69 | Overweighting of small probabilities in loss domain |
    """)


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 NeuroQuant")
    st.markdown("---")
    nav = st.radio("Navigate", ["🏠 Home", "🧪 Experiment", "📊 Results", "📚 Theory"],
                   index=["home", "experiment", "results", "theory"].index(
                       st.session_state.get("page", "home")
                   ))
    nav_map = {
        "🏠 Home":       "home",
        "🧪 Experiment": "experiment",
        "📊 Results":    "results",
        "📚 Theory":     "theory",
    }
    if nav_map[nav] != st.session_state["page"]:
        st.session_state["page"] = nav_map[nav]
        st.rerun()

# ------------------------------------------------------------------
# Render current page
# ------------------------------------------------------------------

page = st.session_state["page"]

if page == "home":
    page_home()
elif page == "experiment":
    if not st.session_state.get("session_started"):
        st.session_state["page"] = "home"
        st.rerun()
    else:
        page_experiment()
elif page == "results":
    page_results()
elif page == "theory":
    page_theory()
