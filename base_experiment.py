"""
experiments/base_experiment.py
--------------------------------
Abstract base class for all NeuroQuant experiments.

Each experiment must define:
  - hypothesis: the testable prediction
  - variables: independent, dependent, and controlled
  - setup(): returns the ordered scenario sequence
  - analyze(): computes the key output metrics

This structure enforces scientific rigor across all experiments.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.engine import SimulationEngine, SimulationResult
from core.scenario import Scenario


# ------------------------------------------------------------------
# Experiment Metadata
# ------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name:           str
    hypothesis:     str
    iv_description: str   # Independent variable
    dv_description: str   # Dependent variable
    n_trials:       int   = 40
    n_simulations:  int   = 200
    seed:           int   = 42
    notes:          str   = ""


@dataclass
class ExperimentResult:
    config:           ExperimentConfig
    raw_data:         pd.DataFrame
    aggregate_metrics: Dict[str, Any]
    hypothesis_supported: Optional[bool] = None
    timestamp:        str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output = {
            "experiment":     self.config.name,
            "hypothesis":     self.config.hypothesis,
            "timestamp":      self.timestamp,
            "metrics":        self.aggregate_metrics,
            "hypothesis_supported": self.hypothesis_supported,
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"EXPERIMENT: {self.config.name}",
            f"{'='*60}",
            f"Hypothesis: {self.config.hypothesis}",
            f"",
            f"Key Metrics:",
        ]
        for k, v in self.aggregate_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<35} {v:.4f}")
            else:
                lines.append(f"  {k:<35} {v}")
        verdict = "SUPPORTED" if self.hypothesis_supported else (
            "NOT SUPPORTED" if self.hypothesis_supported is False else "INCONCLUSIVE"
        )
        lines.append(f"")
        lines.append(f"  Hypothesis: {verdict}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Abstract Experiment
# ------------------------------------------------------------------

class BaseExperiment(ABC):
    """
    Abstract experiment. Subclasses implement setup() and analyze().
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.engine = SimulationEngine(seed=config.seed)

    @property
    @abstractmethod
    def scenarios(self) -> List[Scenario]:
        """Return the ordered list of scenarios for this experiment."""

    @abstractmethod
    def run_condition(self, condition: str) -> SimulationResult:
        """Run one experimental condition. Returns a SimulationResult."""

    @abstractmethod
    def analyze(self) -> ExperimentResult:
        """Run full experiment and return ExperimentResult."""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def risky_rate(self, df: pd.DataFrame) -> float:
        """Fraction of trials where the risky option was chosen."""
        return float((df["choice"] == "risky").mean())

    def ev_deviation_mean(self, df: pd.DataFrame) -> float:
        """Mean EV left on the table (ev_optimal - ev_chosen)."""
        return float((df["ev_optimal"] - df["ev_chosen"]).mean())

    def optimality_rate(self, df: pd.DataFrame) -> float:
        return float(df["is_optimal"].mean())

    def rolling_risky_rate(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        return (df["choice"] == "risky").rolling(window).mean()
