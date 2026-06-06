"""
Snapshot persistence — save and load full experiment state.

Complements System0Sandbox.save() with a typed loading API so saved runs
can be inspected, compared, or handed to replay_experiment without manual
path construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from metabolic_intelligence_lab.core.agent import DUSEAgent
from metabolic_intelligence_lab.core.memory import GeometricMemory
from metabolic_intelligence_lab.core.prospective import ProspectiveLog
from metabolic_intelligence_lab.core.salience import SalienceEngine
from metabolic_intelligence_lab.core.ternary_field import TernaryField
from metabolic_intelligence_lab.core.world import WorldState
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig


@dataclass
class RunSnapshot:
    """All persisted state for a single completed run."""

    config: ExperimentConfig
    agent_state: Dict[str, Any]
    summary: Dict[str, Any]
    trace: List[Dict[str, Any]]
    run_dir: Path

    # -- derived helpers -------------------------------------------------------

    @property
    def config_name(self) -> str:
        return self.config.name

    @property
    def n_plans(self) -> int:
        return int(self.summary.get("n_plans", 0))

    @property
    def avg_reward(self) -> float:
        return float(self.summary.get("avg_reward", 0.0))

    def restore_agent(self) -> DUSEAgent:
        """Reconstruct a DUSEAgent from the saved snapshot (read-only; no tools attached)."""
        d = self.agent_state
        agent = DUSEAgent(
            name=str(d.get("name", "restored")),
            dim=int(d.get("dim", 6)),
            salience_threshold=float(d.get("salience_threshold", 1.2)),
            prospective_depth=int(d.get("prospective_depth", 1)),
        )
        if "memory" in d:
            agent.gm = GeometricMemory.from_dict(d["memory"])
        if "salience" in d:
            agent.se = SalienceEngine.from_dict(d["salience"])
        if "plog" in d:
            agent.plog = ProspectiveLog.from_dict(d["plog"])
        if d.get("ternary"):
            agent.ternary_field = TernaryField.from_dict(d["ternary"])
        if "world" in d:
            agent.world = WorldState.from_dict(d["world"])
        return agent


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_snapshot(
    run_dir: Path,
    agent: DUSEAgent,
    config: ExperimentConfig,
    trace: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Path:
    """Write all snapshot artefacts to *run_dir* and return the directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config.save(run_dir / "config.yaml")
    agent.save_json(run_dir / "agent_snapshot.json")

    with (run_dir / "trace.json").open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return run_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_snapshot(run_dir: str | Path) -> RunSnapshot:
    """Load a saved run directory into a :class:`RunSnapshot`."""
    run_dir = Path(run_dir)

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    config = ExperimentConfig.from_yaml(config_path)

    snapshot_path = run_dir / "agent_snapshot.json"
    agent_state: Dict[str, Any] = {}
    if snapshot_path.exists():
        with snapshot_path.open(encoding="utf-8") as f:
            agent_state = json.load(f)

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)

    trace_path = run_dir / "trace.json"
    trace: List[Dict[str, Any]] = []
    if trace_path.exists():
        with trace_path.open(encoding="utf-8") as f:
            trace = json.load(f)

    return RunSnapshot(
        config=config,
        agent_state=agent_state,
        summary=summary,
        trace=trace,
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_runs(results_root: str | Path = "results") -> List[Path]:
    """Return all run directories under *results_root*, sorted newest-first."""
    root = Path(results_root)
    runs: List[Path] = []
    for child in root.rglob("config.yaml"):
        runs.append(child.parent)
    return sorted(runs, reverse=True)


def latest_run(config_name: str, results_root: str | Path = "results") -> Optional[Path]:
    """Return the most recent run directory for *config_name*, or None."""
    root = Path(results_root) / config_name
    if not root.exists():
        return None
    candidates = sorted(
        [d for d in root.iterdir() if d.is_dir() and (d / "config.yaml").exists()],
        reverse=True,
    )
    return candidates[0] if candidates else None
