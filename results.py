from __future__ import annotations
from typing import Any, Dict, List, Optional
from .reward import RewardModel
from .telemetry import Telemetry
from .tools import MockToolAPI
from .world import WorldState


class System2Plugin:
    def __call__(self, agent_name: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError


class RulePlanner(System2Plugin):
    """Deterministic micro-planner with preconditions/effects, branching, tool budget, and reward eval."""

    def __init__(self, tools: MockToolAPI, world: WorldState, log: Telemetry, tool_budget: float = 0.3):
        self.tools = tools
        self.world = world
        self.log = log
        self.tool_budget = tool_budget
        self.tool_spent = 0.0
        self.reward_model = RewardModel()
        self.plan_map: Dict[str, List[str]] = {
            "cold": ["check_weather", "find_shelter", "prepare_warmth_kit", "collect_wood", "ignite_fire"],
            "food": ["scan_inventory", "forage"],
            "fire": ["collect_wood", "ignite_fire"],
            "shelter": ["inspect_shelter", "reinforce_shelter"],
            "tool": ["scan_inventory", "craft_tool"],
        }

    def _pre_ok(self, act: str) -> bool:
        w = self.world
        if act == "ignite_fire":
            return w.has("sticks", 1)
        if act == "reinforce_shelter":
            return w.has("cordage", 1) and w.has("sticks", 1)
        return True

    def _apply_effects(self, act: str, res: Dict[str, Any]) -> None:
        w = self.world
        d = res.get("data", {})
        if act == "check_weather":
            w.weather.update(d)
        elif act == "find_shelter":
            w.shelter["status"] = "located"
            w.shelter["location"] = d.get("location")
        elif act == "prepare_warmth_kit":
            w.add("firestarter", 1)
            w.add("blanket", 1)
        elif act == "scan_inventory":
            for it in d.get("items", []):
                w.add(it, 0)
        elif act == "forage":
            for it in d.get("found", []):
                w.add(it, 1)
        elif act == "collect_wood":
            w.add("sticks", int(d.get("bundles", 0)))
        elif act == "ignite_fire":
            if w.consume("sticks", 1):
                w.fire["lit"] = True
                w.fire["duration_min"] = max(w.fire.get("duration_min", 0), int(d.get("duration_min", 30)))
        elif act == "inspect_shelter":
            w.shelter.update(d)
        elif act == "reinforce_shelter":
            if w.consume("cordage", 1) and w.consume("sticks", 1):
                w.shelter.update(d)
        elif act == "craft_tool":
            w.add("stone_axe", 1)

    def _branch(self, goal: str, act: str) -> Optional[List[str]]:
        w = self.world
        if goal == "cold" and act == "check_weather" and w.weather.get("status") == "mild":
            return ["scan_inventory"]
        if act == "ignite_fire" and not self._pre_ok("ignite_fire"):
            return ["collect_wood", "ignite_fire"]
        if act == "reinforce_shelter" and not self._pre_ok("reinforce_shelter"):
            return ["forage", "craft_tool", "reinforce_shelter"]
        return None

    def __call__(self, agent_name: str, data: Dict[str, Any]) -> None:
        obs = data.get("obs", {})
        goal = data.get("goal") or (obs.get("top") or "reflect")
        plan = list(self.plan_map.get(goal, []))
        sdf_depth = int(data.get("sdf_depth", 1))
        energy_used = float(data.get("energy_used", 0.0))
        print(f"[Planner] {agent_name}: goal={goal} plan={plan} budget={self.tool_budget:.2f} depth={sdf_depth}")
        self.tool_spent = 0.0
        total_cost, total_lat = 0.0, 0
        step = 0
        while step < len(plan):
            act = plan[step]
            est_cost = 0.02
            if self.tool_spent + est_cost > self.tool_budget:
                print(f"   - budget exhausted ({self.tool_spent:.2f}/{self.tool_budget:.2f}); stopping plan")
                break

            repl = self._branch(goal, act)
            if repl is not None:
                print(f"   - branch: {act} → {repl}")
                plan[step:step + 1] = repl
                continue

            if not self._pre_ok(act):
                print(f"   - precondition failed for {act}; replanning…")
                plan[step:step + 1] = (self._branch(goal, act) or [])
                continue

            tool_fn = getattr(self.tools, act, None)
            if callable(tool_fn):
                res = tool_fn()
                self._apply_effects(act, res)
                cost = float(res.get("cost", 0.0))
                lat = int(res.get("latency_ms", 0))
                self.tool_spent += cost
                total_cost += cost
                total_lat += lat
                print(f"   - tool:{res['tool']} ok={res['ok']} cost={cost:.2f} lat={lat} data={res['data']}")
                self.log.log(
                    t=obs.get("t", -1),
                    agent_name=agent_name,
                    tag=f"plan:{goal}:{act}",
                    extra={
                        "tool": res["tool"],
                        "cost": cost,
                        "latency_ms": lat,
                        "tool_budget": self.tool_budget,
                        "tool_spent": self.tool_spent,
                        "world": self.world.to_dict(),
                    },
                )
            else:
                print(f"   - missing tool: {act}")
            step += 1

        if plan:
            metrics = self.reward_model.eval(
                world=self.world,
                energy_used=energy_used,
                tool_cost=total_cost,
                latency_ms=total_lat,
                depth=sdf_depth,
            )
            print(
                f"   → total_cost={total_cost:.2f}, total_latency_ms={total_lat}, "
                f"reward={metrics['total_reward']:.3f}, ces_like={metrics['ces_like']:.3f}"
            )
            self.log.log(t=obs.get("t", -1), agent_name=agent_name, tag=f"plan_reward:{goal}", extra=metrics)


class LLMStub(System2Plugin):
    def __call__(self, agent_name: str, data: Dict[str, Any]) -> None:
        print(f"[LLMStub] {agent_name}: {data.get('topic')} | reasoning burst invoked")
