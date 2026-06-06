from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ExperimentResult:
    config_name: str
    avg_reward: float
    avg_ces_like: float
    avg_energy: float
    avg_latency: float
    stability_score: float
    entropy_score: float
    frontier_variance: float
    survival_rate: float
    n_plans: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_rows(config_name: str, rows: List[Dict[str, Any]]) -> ExperimentResult:
    plan_rows = [r for r in rows if str(r.get("tag", "")).startswith("plan_reward:")]
    tick_rows = [r for r in rows if r.get("tag") == "tick"]

    n_plans = len(plan_rows)
    avg_reward = sum(float(r.get("total_reward", 0.0)) for r in plan_rows) / n_plans if n_plans else 0.0
    avg_ces = sum(float(r.get("ces_like", 0.0)) for r in plan_rows) / n_plans if n_plans else 0.0
    avg_latency = sum(float(r.get("time_cost", 0.0)) for r in plan_rows) / n_plans if n_plans else 0.0
    avg_energy = sum(float(r.get("energy_used", 0.0)) for r in tick_rows) / len(tick_rows) if tick_rows else 0.0

    thresholds = [float(r.get("threshold", 0.0)) for r in tick_rows if "threshold" in r]
    stability = 1.0 / (1.0 + (max(thresholds) - min(thresholds))) if thresholds else 0.0

    labels = [str(r.get("top_label")) for r in tick_rows if r.get("top_label")]
    if labels:
        from math import log
        counts = {x: labels.count(x) for x in set(labels)}
        total = len(labels)
        entropy = -sum((c / total) * log(c / total + 1e-12) for c in counts.values())
    else:
        entropy = 0.0

    frontier_sizes = [len(r.get("frontier", [])) for r in tick_rows if isinstance(r.get("frontier", []), list)]
    if frontier_sizes:
        m = sum(frontier_sizes) / len(frontier_sizes)
        frontier_variance = sum((x - m) ** 2 for x in frontier_sizes) / len(frontier_sizes)
    else:
        frontier_variance = 0.0

    survival_values = [float(r.get("survival", 0.0)) for r in plan_rows if "survival" in r]
    survival_rate = sum(1.0 for x in survival_values if x >= 0.5) / len(survival_values) if survival_values else 0.0

    return ExperimentResult(
        config_name=config_name,
        avg_reward=avg_reward,
        avg_ces_like=avg_ces,
        avg_energy=avg_energy,
        avg_latency=avg_latency,
        stability_score=stability,
        entropy_score=entropy,
        frontier_variance=frontier_variance,
        survival_rate=survival_rate,
        n_plans=n_plans,
    )
