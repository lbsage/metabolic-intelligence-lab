from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

from metabolic_intelligence_lab.core.agent import DUSEAgent
from metabolic_intelligence_lab.core.planner import RulePlanner
from metabolic_intelligence_lab.core.reward import RewardModel


@dataclass
class ExperimentConfig:
    name: str
    salience_threshold: float = 1.2
    reserve: float = 0.3
    gamma_cost: float = 0.7
    delta_time: float = 0.3
    tool_budget: float = 0.3
    coupling_gamma: float = 0.02
    coupling_decay: float = 0.01
    prospective_depth: int = 1
    memory_decay: float = 0.01
    seed: int = 42
    steps: int = 30
    visualize: bool = False

    def __post_init__(self):
        if not (0.0 <= self.salience_threshold <= 2.0):
            raise ValueError("salience_threshold must be between 0 and 2")
        if not (0.0 <= self.reserve <= 0.95):
            raise ValueError("reserve must be between 0 and 0.95")
        if self.tool_budget < 0:
            raise ValueError("tool_budget must be non-negative")
        if self.prospective_depth < 1:
            raise ValueError("prospective_depth must be >= 1")
        if not (0.0 <= self.memory_decay < 1.0):
            raise ValueError("memory_decay must be in [0, 1)")

    def apply_to(self, agent: DUSEAgent, planner: RulePlanner) -> None:
        agent.salience_threshold = self.salience_threshold
        agent.energy.reserve = self.reserve
        agent.prospective_depth = self.prospective_depth
        agent.gm.memory_decay = self.memory_decay
        planner.tool_budget = self.tool_budget
        planner.reward_model = RewardModel(gamma_cost=self.gamma_cost, delta_time=self.delta_time)
        if agent.ternary_field:
            agent.ternary_field.set_coupling_dynamics(self.coupling_gamma, self.coupling_decay)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return ExperimentConfig(**data)
