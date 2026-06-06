"""
System 0 Sandbox — canonical entry point for the metabolic intelligence framework.

Wraps DUSEAgent with:
- World state evolution (hunger, fire burnout, weather drift, shelter decay)
- Semantically grounded context vectors derived from world state, so that
  similar survival situations retrieve similar past memories
- Named cognitive phase trace per tick, logged to telemetry and self.trace
- Clean run(steps) / step(t) API
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from metabolic_intelligence_lab.core.agent import DUSEAgent
from metabolic_intelligence_lab.core.bus import DUSEBus
from metabolic_intelligence_lab.core.planner import LLMStub, RulePlanner
from metabolic_intelligence_lab.core.tools import MockToolAPI
from metabolic_intelligence_lab.core.world import WorldState
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.results import summarize_rows

LABELS = ["food", "shelter", "fire", "cold", "tool"]


def world_to_context(
    world: WorldState,
    dim: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Encode world state as a latent context vector for geometric memory retrieval.

    Each of the first six dimensions captures a survival-relevant signal so that
    similar world states produce similar vectors — enabling memory to retrieve
    analogous past experiences rather than random noise.
    """
    rng = rng or np.random.default_rng()
    temp = float(world.weather.get("temp_c", 10))
    temp_norm = float(np.clip((temp + 5) / 20.0, 0.0, 1.0))       # -5..15 → 0..1
    hunger_norm = float(np.clip(world.hunger / 20.0, 0.0, 1.0))
    fire_lit = 1.0 if world.fire.get("lit", False) else 0.0
    leaks = int(world.shelter.get("leaks", 0))
    shelter_norm = float(np.clip(1.0 - leaks / 4.0, 0.0, 1.0))
    has_sticks = 1.0 if world.has("sticks", 1) else 0.0
    has_food = 1.0 if (world.has("berries", 1) or world.has("mushrooms", 1)) else 0.0

    base = np.array(
        [temp_norm, hunger_norm, fire_lit, shelter_norm, has_sticks, has_food],
        dtype=float,
    )
    if dim <= 6:
        return base[:dim]
    return np.concatenate([base, rng.random(dim - 6) * 0.1])


def evolve_world(world: WorldState, rng: np.random.Generator) -> Dict[str, Any]:
    """Advance world state one step — environment dynamics that run regardless of the agent."""
    events: Dict[str, Any] = {}

    world.hunger = min(20, world.hunger + 1)
    if world.hunger >= 10:
        events["hunger_warning"] = world.hunger

    if world.fire.get("lit", False):
        dur = max(0, int(world.fire.get("duration_min", 0)) - 5)
        world.fire["duration_min"] = dur
        if dur <= 0:
            world.fire["lit"] = False
            events["fire_out"] = True

    temp = float(world.weather.get("temp_c", 10))
    temp = float(np.clip(temp + int(rng.integers(-1, 2)), -5, 15))
    world.weather["temp_c"] = temp
    world.weather["status"] = "cold" if temp < 8 else "mild"
    if temp < 0:
        events["temp_critical"] = temp

    if rng.random() < 0.05:
        leaks = min(3, int(world.shelter.get("leaks", 0)) + 1)
        world.shelter["leaks"] = leaks
        events["shelter_degraded"] = leaks

    return events


class System0Sandbox:
    """
    Orchestrates the full System 0 cognitive loop over an agent embedded in a
    dynamic environment.

    Cognitive phases per tick:
      1. evolve     — environment dynamics advance (hunger, fire, weather, shelter)
      2. sense      — world-state context vector → geometric memory retrieval
      3. modulate   — ternary field adjusts salience weights (inside agent.act)
      4. arbitrate  — goal utility frontier selects best goal
      5. gate       — escalate to System 2 when salience + energy threshold met
      6. learn      — prospective log update, memory reinforce, policy adapt
      7. emit       — structured phase trace appended to self.trace and telemetry

    Usage::

        cfg = ExperimentConfig(name="baseline", steps=30)
        sb = System0Sandbox(cfg)
        sb.run()
        sb.save("results/my_run")
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        name: str = "scout",
        shared_bus: Optional[DUSEBus] = None,
    ):
        self.cfg = cfg
        self.name = name
        self._rng = np.random.default_rng(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.bus: DUSEBus = shared_bus or DUSEBus()
        self.tools = MockToolAPI()
        self.agent = self._build_agent()
        self.bus.register(self.agent.id)
        self.trace: List[Dict[str, Any]] = []

    def _build_agent(self) -> DUSEAgent:
        agent = DUSEAgent(name=self.name, salience_threshold=self.cfg.salience_threshold)
        agent.seed(LABELS)
        planner = RulePlanner(self.tools, agent.world, agent.log, tool_budget=self.cfg.tool_budget)
        agent.system2_plugins = [planner, LLMStub()]
        self.cfg.apply_to(agent, planner)
        return agent

    def step(self, t: int) -> Dict[str, Any]:
        agent = self.agent

        # Reset per-tick energy budget
        agent.energy.reset()

        # 1. Evolve world
        env_events = evolve_world(agent.world, self._rng)

        # 2. Sense
        ctx = world_to_context(agent.world, dim=agent.dim, rng=self._rng)
        obs = agent.sense(t, ctx)

        # 3+4+5. Modulate + Arbitrate + Gate
        frontier = agent.goal_frontier()
        goal, goal_score = agent.choose_goal()
        will_escalate = bool(
            goal and goal_score > agent.salience_threshold and agent.energy.can_spend(0.25)
        )
        field_state = dict(agent.ternary_field.states) if agent.ternary_field else {}
        agent.act(obs, bus=self.bus)

        # 6. Learn
        agent.learn(obs)

        # Fire any due prospective tasks
        for task in agent.tasks.due(t, agent.se.top(1)):
            print(f"[Task] {agent.name}: execute {task.action} (id={task.id}) at t={t}")

        # Update viz histories on agent
        agent._viz_threshold_hist.append(agent.salience_threshold)
        agent._viz_salience_hist.append(agent.se.top(1)[0][1] if agent.se.top(1) else 0.0)
        if agent.ternary_field:
            agent._viz_field_state_hist.append(dict(agent.ternary_field.states))

        # Log tick row so summarize_rows and Telemetry.summary_stats work correctly
        agent.log.log(
            t=t,
            agent_name=agent.name,
            tag="tick",
            extra={
                "energy_used": round(agent.energy.used, 4),
                "threshold": round(agent.salience_threshold, 4),
                "reserve": round(agent.energy.reserve, 4),
                "top_salience": agent.se.top(1)[0][1] if agent.se.top(1) else 0.0,
                "top_label": agent.se.top(1)[0][0] if agent.se.top(1) else None,
                "frontier": frontier,
                "world": agent.world.to_dict(),
            },
        )

        # Drain bus
        self.bus.recv(agent.id)

        # 7. Emit
        tick: Dict[str, Any] = {
            "t": t,
            "env": env_events,
            "sense": {"top_label": obs.get("top"), "forecast": obs.get("forecast", [])},
            "field": field_state,
            "frontier": frontier,
            "goal": goal,
            "goal_score": round(float(goal_score), 4),
            "escalated": will_escalate,
            "energy_used": round(agent.energy.used, 4),
            "threshold": round(agent.salience_threshold, 4),
            "reserve": round(agent.energy.reserve, 4),
            "top_salience": agent.se.top(1)[0][1] if agent.se.top(1) else 0.0,
            "world": agent.world.to_dict(),
        }
        self.trace.append(tick)
        return tick

    def run(
        self,
        steps: Optional[int] = None,
        step_callback: Optional[Callable[[int, Dict[str, Any], DUSEAgent], None]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the sandbox for ``steps`` ticks and return the full trace."""
        n = steps if steps is not None else self.cfg.steps
        for t in range(n):
            tick = self.step(t)
            if step_callback:
                step_callback(t, tick, self.agent)
            if verbose:
                label = tick["sense"]["top_label"] or "—"
                goal_str = (
                    f"→{tick['goal']}({tick['goal_score']:.2f})" if tick["goal"] else ""
                )
                esc = " [S2]" if tick["escalated"] else ""
                env_str = (
                    "  " + " ".join(f"[{k}]" for k in tick["env"]) if tick["env"] else ""
                )
                print(
                    f"[{self.name} t={t:02d}] {label:<8} "
                    f"E={tick['energy_used']:.2f} thr={tick['threshold']:.2f} "
                    f"{goal_str}{esc}{env_str}"
                )
        return self.trace

    def save(self, out_dir: str | Path) -> Path:
        """Persist config, agent snapshot, telemetry, trace, and summary to ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.cfg.save(out / "config.yaml")
        self.agent.save_json(out / "agent_snapshot.json")
        self.agent.log.write_json(out / "telemetry.json")
        self.agent.log.write_csv(out / "telemetry.csv")
        with (out / "trace.json").open("w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, default=str)
        with (out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
        return out

    @property
    def summary(self) -> Dict[str, Any]:
        return summarize_rows(self.cfg.name, self.agent.log.rows).to_dict()

    @staticmethod
    def run_id() -> str:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
