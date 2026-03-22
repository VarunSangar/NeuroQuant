"""
data/session_store.py
---------------------
Lightweight session storage for decision logging and persistence.

Decisions are logged incrementally and saved as CSV + JSON at session end.
Designed to work both for batch simulations and live UI sessions.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


# ------------------------------------------------------------------
# Decision record
# ------------------------------------------------------------------

@dataclass
class DecisionRecord:
    """One logged decision in a session."""
    session_id:     str
    trial_index:    int
    timestamp:      float           # Unix timestamp
    scenario_id:    str
    frame:          str
    choice:         str             # 'risky' or 'safe'
    outcome:        float
    ev_chosen:      float
    ev_optimal:     float
    is_optimal:     bool
    time_to_decide: Optional[float] = None   # Seconds (from UI)
    cumulative_pnl: float           = 0.0
    streak:         int             = 0
    agent_type:     str             = "human"
    metadata:       dict            = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dt"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------

class Session:
    """
    A single experimental session (one participant or simulation run).

    Collects decisions in memory, provides access to running metrics,
    and persists data to disk at session end.
    """

    def __init__(
        self,
        session_id:  Optional[str] = None,
        agent_type:  str = "human",
        output_dir:  str = "results",
    ):
        self.session_id  = session_id or str(uuid.uuid4())[:12]
        self.agent_type  = agent_type
        self.output_dir  = output_dir
        self._decisions: List[DecisionRecord] = []
        self._start_time = time.time()
        self._last_time  = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        trial_index:  int,
        scenario_id:  str,
        frame:        str,
        choice:       str,
        outcome:      float,
        ev_chosen:    float,
        ev_optimal:   float,
        is_optimal:   bool,
        streak:       int = 0,
        metadata:     dict = None,
    ) -> DecisionRecord:
        """Log a single decision."""
        now            = time.time()
        ttd            = now - self._last_time if self._decisions else None
        self._last_time = now

        pnl = (self._decisions[-1].cumulative_pnl if self._decisions else 0.0) + outcome

        record = DecisionRecord(
            session_id     = self.session_id,
            trial_index    = trial_index,
            timestamp      = now,
            scenario_id    = scenario_id,
            frame          = frame,
            choice         = choice,
            outcome        = outcome,
            ev_chosen      = ev_chosen,
            ev_optimal     = ev_optimal,
            is_optimal     = is_optimal,
            time_to_decide = ttd,
            cumulative_pnl = pnl,
            streak         = streak,
            agent_type     = self.agent_type,
            metadata       = metadata or {},
        )
        self._decisions.append(record)
        return record

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        rows = [d.to_dict() for d in self._decisions]
        return pd.DataFrame(rows)

    @property
    def n_trials(self) -> int:
        return len(self._decisions)

    @property
    def total_pnl(self) -> float:
        return self._decisions[-1].cumulative_pnl if self._decisions else 0.0

    @property
    def optimality_rate(self) -> float:
        if not self._decisions:
            return 0.0
        return sum(1 for d in self._decisions if d.is_optimal) / len(self._decisions)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Dict[str, str]:
        """Save session to CSV and JSON. Returns dict of file paths."""
        os.makedirs(self.output_dir, exist_ok=True)
        prefix    = os.path.join(self.output_dir, f"session_{self.session_id}")

        # CSV
        csv_path  = f"{prefix}_decisions.csv"
        self.to_dataframe().to_csv(csv_path, index=False)

        # JSON summary
        json_path = f"{prefix}_summary.json"
        summary   = {
            "session_id":    self.session_id,
            "agent_type":    self.agent_type,
            "n_trials":      self.n_trials,
            "total_pnl":     round(self.total_pnl, 2),
            "optimality_rate": round(self.optimality_rate, 4),
            "duration_sec":  round(time.time() - self._start_time, 1),
            "timestamp":     datetime.now().isoformat(),
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {"csv": csv_path, "json": json_path}

    @classmethod
    def load(cls, csv_path: str) -> "Session":
        """Reconstruct a session from a saved CSV."""
        df       = pd.read_csv(csv_path)
        session  = cls(
            session_id = str(df["session_id"].iloc[0]),
            agent_type = str(df["agent_type"].iloc[0]),
        )
        for _, row in df.iterrows():
            session.record(
                trial_index = int(row["trial_index"]),
                scenario_id = str(row["scenario_id"]),
                frame       = str(row["frame"]),
                choice      = str(row["choice"]),
                outcome     = float(row["outcome"]),
                ev_chosen   = float(row["ev_chosen"]),
                ev_optimal  = float(row["ev_optimal"]),
                is_optimal  = bool(row["is_optimal"]),
                streak      = int(row.get("streak", 0)),
            )
        return session


# ------------------------------------------------------------------
# Session Manager (multi-session storage)
# ------------------------------------------------------------------

class SessionManager:
    """Manages a collection of sessions for multi-participant analysis."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self._sessions: Dict[str, Session] = {}

    def new_session(self, agent_type: str = "human") -> Session:
        s = Session(agent_type=agent_type, output_dir=self.output_dir)
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def all_decisions(self) -> pd.DataFrame:
        """Concatenate all sessions' decisions into one DataFrame."""
        dfs = [s.to_dataframe() for s in self._sessions.values() if s.n_trials > 0]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def save_all(self) -> List[Dict[str, str]]:
        return [s.save() for s in self._sessions.values()]
