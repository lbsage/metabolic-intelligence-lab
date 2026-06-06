from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


class Telemetry:
    """Step-level CSV/JSON logger for agent and planner activity."""

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def log(self, t: int, agent_name: str, tag: str, extra: Optional[Dict[str, Any]] = None) -> None:
        row: Dict[str, Any] = {"t": t, "agent": agent_name, "tag": tag}
        if extra:
            row.update(extra)
        self.rows.append(row)

    # ------------------------------------------------------------------ query

    def filter_by_tag(self, tag: str, prefix: bool = True) -> List[Dict[str, Any]]:
        """Return rows whose tag starts with (or exactly matches) ``tag``."""
        if prefix:
            return [r for r in self.rows if str(r.get("tag", "")).startswith(tag)]
        return [r for r in self.rows if r.get("tag") == tag]

    def summary_stats(self) -> Dict[str, Any]:
        """Compute inline metrics without requiring ExperimentResult."""
        plan_rows = self.filter_by_tag("plan_reward:")
        tick_rows = self.filter_by_tag("tick", prefix=False)

        n_plans = len(plan_rows)
        avg_reward = (
            sum(float(r.get("total_reward", 0.0)) for r in plan_rows) / n_plans
            if n_plans else 0.0
        )
        avg_ces = (
            sum(float(r.get("ces_like", 0.0)) for r in plan_rows) / n_plans
            if n_plans else 0.0
        )
        avg_energy = (
            sum(float(r.get("energy_used", 0.0)) for r in tick_rows) / len(tick_rows)
            if tick_rows else 0.0
        )

        top_labels = [str(r["top_label"]) for r in tick_rows if r.get("top_label")]
        label_dist = dict(Counter(top_labels))

        goals_hit = [str(r["tag"]).split("plan_reward:")[-1] for r in plan_rows]
        goal_dist = dict(Counter(goals_hit))

        survival_vals = [float(r["survival"]) for r in plan_rows if "survival" in r]
        survival_rate = (
            sum(1 for x in survival_vals if x >= 0.5) / len(survival_vals)
            if survival_vals else 0.0
        )

        return {
            "n_ticks": len(tick_rows),
            "n_plans": n_plans,
            "avg_reward": round(avg_reward, 4),
            "avg_ces": round(avg_ces, 4),
            "avg_energy": round(avg_energy, 4),
            "survival_rate": round(survival_rate, 4),
            "label_distribution": label_dist,
            "goal_distribution": goal_dist,
        }

    # ----------------------------------------------------------------- write

    def write_csv(self, path: str | Path) -> None:
        if not self.rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in self.rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2, default=str)

    def write_replay(self, path: str | Path, config_name: str, seed: int) -> None:
        """Write a replay manifest so replay_experiment can re-run this episode deterministically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config_name": config_name,
            "seed": seed,
            "n_rows": len(self.rows),
            "stats": self.summary_stats(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
