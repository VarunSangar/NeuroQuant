# NeuroQuant
### Modeling Human Decision-Making Under Uncertainty in Simulated Trading Environments

---

## Concept

NeuroQuant is an interdisciplinary research framework that sits at the intersection of **quantitative finance**, **behavioral neuroscience**, and **experimental psychology**. It provides:

1. A **simulation engine** for generating trading scenarios with configurable probability distributions, expected values, and volatility profiles.
2. A **psychological modeling layer** implementing Kahneman-Tversky prospect theory, loss aversion, recency bias, and decision fatigue.
3. A **controlled experimental framework** for running structured behavioral experiments on human decision-making.
4. An **adaptive AI agent** whose behavior can be compared against human participants.
5. A **data collection and analysis pipeline** that quantifies deviation from rational (EV-maximizing) strategy.

This project is designed to answer: *How and when do humans deviate from optimal decision-making, and what cognitive mechanisms drive those deviations?*

---

## Project Structure

```
neuroquant/
├── core/
│   ├── scenario.py          # Trade scenario definitions and probability models
│   ├── engine.py            # Monte Carlo simulation engine
│   └── strategy.py          # Baseline strategy implementations (rational, risk-averse, risk-seeking)
├── psychology/
│   ├── prospect_theory.py   # Kahneman-Tversky value/weighting functions
│   ├── biases.py            # Loss aversion, recency bias, decision fatigue
│   └── behavioral_model.py  # Composite behavioral agent
├── experiments/
│   ├── base_experiment.py   # Abstract experiment class
│   ├── exp1_framing.py      # Experiment 1: Gain vs Loss Framing
│   └── exp2_streaks.py      # Experiment 2: Streak-Induced Behavior
├── agents/
│   ├── base_agent.py        # Abstract agent interface
│   ├── human_agent.py       # Human decision recording wrapper
│   └── adaptive_agent.py    # Heuristic adaptive AI agent
├── analysis/
│   ├── metrics.py           # EV deviation, behavioral shift metrics
│   └── statistics.py        # Hypothesis testing, correlation analysis
├── visualization/
│   └── plots.py             # All visualization functions
├── data/
│   └── session_store.py     # Decision logging and persistence
├── ui/
│   └── app.py               # Streamlit interactive UI
├── sample_data/
│   └── generate_sample.py   # Generates synthetic human-like sample data
├── tests/
│   └── test_core.py         # Unit tests
├── run_experiment.py        # CLI experiment runner
├── run_analysis.py          # CLI analysis runner
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone or download the project
cd neuroquant

# Install dependencies
pip install -r requirements.txt

# Run sample data generation
python sample_data/generate_sample.py

# Run experiment simulation
python run_experiment.py --experiment framing --trials 40 --mode simulate

# Run analysis on collected data
python run_analysis.py --session results/latest_session.json

# Launch interactive UI
streamlit run ui/app.py
```

---

## Experiments

### Experiment 1: Gain vs Loss Framing
**Hypothesis:** Participants will exhibit greater risk-aversion when choices are framed as gains and greater risk-seeking when framed as losses, even when the underlying expected values are identical.

- **Independent Variable:** Frame (gain vs. loss)
- **Dependent Variable:** Proportion of risky choices
- **Control:** Identical probability distributions and EV across frames

### Experiment 2: Streak-Induced Behavior
**Hypothesis:** Participants exposed to artificial win streaks will over-estimate future win probability (hot-hand fallacy), while loss streaks will induce excessive risk-aversion or paradoxical risk-seeking (gambler's fallacy).

- **Independent Variable:** Streak type (win/loss) and length (3/5/7)
- **Dependent Variable:** Deviation from EV-maximizing strategy
- **Control:** Post-streak scenarios with identical EV

---

## Key Concepts

**Expected Value (EV):** `EV = Σ p_i * outcome_i`

**Prospect Theory Value Function:** `V(x) = x^α if x ≥ 0, else -λ(-x)^β`
- α, β ∈ (0,1) capture diminishing sensitivity
- λ > 1 captures loss aversion (typically λ ≈ 2.25)

**Decision Fatigue:** Modeled as a time-decaying rationality parameter that reduces optimal-choice probability after sustained engagement.

**Recency Bias:** Over-weighting of the last N outcomes relative to true probability, modeled as a Bayesian prior update with high recency weight.

---

## Output

Each session produces:
- `decisions.csv` — full decision log with timestamps, choices, outcomes, EV
- `metrics.json` — aggregate behavioral metrics
- `plots/` — equity curves, risk-taking trajectories, deviation plots

---

## Dependencies
- Python 3.9+
- numpy, pandas, scipy
- matplotlib, plotly
- streamlit (for UI)
- rich (for CLI output)
