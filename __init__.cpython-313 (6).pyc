from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid
import json
from pathlib import Path
import numpy as np

from .bus import DUSEBus, EnergyBudget, Message
from .memory import GeometricMemory
from .policy import PolicyModule
from .prospective import ProspectiveEngine, ProspectiveLog
from .salience import SalienceEngine
from .tasks import TaskQueue
from .telemetry import Telemetry
from .ternary_field import TernaryField
from .world import WorldState
from .planner import RulePlanner, System2Plugin


@dataclass
class DUSEAgent:
    name: str
    dim: int = 6
    salience_threshold: float = 1.2
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gm: GeometricMemory = field(default_factory=lambda: GeometricMemory(dim=6))
    se: SalienceEngine = field(default_factory=SalienceEngine)
    pe: Optional[ProspectiveEngine] = None
    plog: ProspectiveLog = field(default_factory=ProspectiveLog)
    policy: PolicyModule = field(default_factory=PolicyModule)
    energy: EnergyBudget = field(default_factory=EnergyBudget)
    system2_plugins: List[System2Plugin] = field(default_factory=list)
    ternary_field: Optional[TernaryField] = None
    tasks: TaskQueue = field(default_factory=TaskQueue)
    world: WorldState = field(default_factory=WorldState)
    log: Telemetry = field(default_factory=Telemetry)
    prospective_depth: int = 1

    _viz_salience_hist: List[float] = field(default_factory=list)
    _viz_threshold_hist: List[float] = field(default_factory=list)
    _viz_field_state_hist: List[Dict[str, int]] = field(default_factory=list)

    def __post_init__(self):
        self.pe = self.pe or ProspectiveEngine(self.gm)

    def seed(self, labels: List[str]) -> None:
        for lb in labels:
            self.gm.store(np.random.rand(self.dim), lb, strength=1.0)
        self.ternary_field = TernaryField(labels)

    def add_prospective_task(self, now: int, delay: int, label_if: Optional[str], min_salience: float, action: str) -> str:
        return self.tasks.add(trigger_time=now + delay, label_if=label_if, min_salience=min_salience, action=action)

    def _goal_utility(self, label: str, salience: float) -> float:
        score = salience
        w = self.world
        if label == "cold":
            temp = w.weather.get("temp_c", 10)
            score += 0.7 if temp < 0 else 0.5 if temp < 5 else 0.3 if temp < 8 else 0.0
        elif label == "food":
            score += 0.1 * float(w.hunger)
        elif label == "shelter":
            leaks = w.shelter.get("leaks", 0)
            score += 0.4 if leaks >= 2 else 0.2 if leaks == 1 else 0.0
        elif label == "fire":
            temp = w.weather.get("temp_c", 10)
            if temp < 8 and not w.fire.get("lit", False):
                score += 0.5
        elif label == "tool":
            if not w.has("stone_axe", 1):
                score += 0.15
        return score

    def goal_frontier(self) -> List[Dict[str, Any]]:
        candidates = self.se.top(5)
        if not candidates:
            return []
        pts: List[Dict[str, Any]] = []
        for label, sal in candidates:
            util = self._goal_utility(label, sal)
            pts.append({"goal": label, "urgency": float(sal), "value": float(util - sal), "utility": float(util)})

        frontier: List[Dict[str, Any]] = []
        for i, a in enumerate(pts):
            dominated = False
            for j, b in enumerate(pts):
                if i == j:
                    continue
                if (
                    b["urgency"] >= a["urgency"]
                    and b["value"] >= a["value"]
                    and (b["urgency"] > a["urgency"] or b["value"] > a["value"])
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(a)
        return sorted(frontier, key=lambda x: x["utility"], reverse=True)

    def choose_goal(self) -> Tuple[Optional[str], float]:
        frontier = self.goal_frontier()
        if not frontier:
            return None, 0.0
        best = frontier[0]
        return best["goal"], best["utility"]

    def sense(self, t: int, context_vec: np.ndarray) -> Dict[str, Any]:
        self.energy.spend(0.05)
        self.gm.decay()
        self.se.decay(rate=0.01)
        matches = self.gm.retrieve(context_vec)
        if matches:
            label = matches[0][1]
            self.se.observe(label)
            forecast = self.pe.simulate(context_vec)
            eid = self.plog.record(t, context_id=self.name, forecast=forecast)
            return {"top": label, "forecast": forecast, "matches": matches, "eid": eid, "t": t}
        return {"top": None, "forecast": [], "matches": [], "eid": None, "t": t}

    def act(self, observation: Dict[str, Any], bus: Optional[DUSEBus] = None) -> None:
        if self.ternary_field:
            sal = self.se.weights
            self.ternary_field.step({k: sal.get(k, 0.0) for k in sal})
            self.se.weights = self.ternary_field.modulate(self.se.weights, gain=0.08)

        goal, goal_score = self.choose_goal()
        top = self.se.top(1)
        top_val = top[0][1] if top else 0.0

        if goal and goal_score > self.salience_threshold:
            if self.energy.spend(0.25):
                topic = f"reason:{goal}"
                available_energy = max(0.0, self.energy.total - self.energy.used - self.energy.reserve)
                tool_budget = max(0.02, available_energy * 0.8)
                for plugin in self.system2_plugins:
                    if isinstance(plugin, RulePlanner):
                        plugin.world = self.world
                        plugin.log = self.log
                        plugin.tool_budget = tool_budget
                    plugin(
                        self.name,
                        {
                            "topic": topic,
                            "obs": observation,
                            "agent": self.name,
                            "goal": goal,
                            "goal_score": goal_score,
                            "top_salience": top_val,
                            "tool_budget": tool_budget,
                            "energy_used": self.energy.used,
                            "sdf_depth": self.prospective_depth,
                        },
                    )
                if bus:
                    bus.send(Message(src=self.id, dst="*", topic=topic, payload={"agent": self.name, "obs": observation, "goal": goal}, priority=0.9))
        else:
            if bus and self.energy.spend(0.02):
                bus.send(Message(src=self.id, dst="*", topic="heartbeat", payload={"agent": self.name, "salience": self.se.top(3)}, priority=0.2))

    def learn(self, observation: Dict[str, Any]) -> None:
        eid = observation.get("eid")
        realized = observation.get("top")
        if eid is not None:
            _ = self.plog.observe(eid, [realized] if realized else [])
            if realized:
                self.gm.reinforce([realized], alpha=0.06, beta=0.01)
                if self.ternary_field:
                    self.ternary_field.reinforce_couplings([realized])

        new_thresh, new_res = self.policy.update(
            last_matches=observation.get("matches", []),
            current_threshold=self.salience_threshold,
            energy_reserve=self.energy.reserve,
        )
        self.salience_threshold = new_thresh
        self.energy.reserve = new_res

    def tick(self, t: int, context_vec: Optional[np.ndarray] = None, bus: Optional[DUSEBus] = None) -> Dict[str, Any]:
        self.energy.reset()
        if context_vec is None:
            context_vec = np.random.rand(self.dim)
        obs = self.sense(t, context_vec)
        self.act(obs, bus=bus)
        self.learn(obs)

        for task in self.tasks.due(t, self.se.top(1)):
            print(f"[Task] {self.name}: execute {task.action} (task_id={task.id}) at t={t}")

        frontier = self.goal_frontier()
        self._viz_threshold_hist.append(self.salience_threshold)
        self._viz_salience_hist.append(self.se.top(1)[0][1] if self.se.top(1) else 0.0)
        if self.ternary_field:
            self._viz_field_state_hist.append(dict(self.ternary_field.states))

        self.log.log(
            t=t,
            agent_name=self.name,
            tag="tick",
            extra={
                "energy_used": round(self.energy.used, 4),
                "threshold": round(self.salience_threshold, 4),
                "reserve": round(self.energy.reserve, 4),
                "top_salience": self.se.top(1)[0][1] if self.se.top(1) else 0.0,
                "top_label": self.se.top(1)[0][0] if self.se.top(1) else None,
                "frontier": frontier,
                "world": self.world.to_dict(),
            },
        )
        return {"energy_used": self.energy.used, "salience": self.se.top(3), "obs": obs, "threshold": self.salience_threshold, "reserve": self.energy.reserve}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dim": self.dim,
            "salience_threshold": self.salience_threshold,
            "prospective_depth": self.prospective_depth,
            "energy": {"total": self.energy.total, "reserve": self.energy.reserve},
            "memory": self.gm.to_dict(),
            "salience": self.se.to_dict(),
            "plog": self.plog.to_dict(),
            "ternary": self.ternary_field.to_dict() if self.ternary_field else None,
            "world": self.world.to_dict(),
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
