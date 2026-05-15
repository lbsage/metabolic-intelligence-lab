from __future__ import annotations
from typing import Any, Dict
import numpy as np


class MockToolAPI:
    """Mock external tools with cost/latency accounting."""

    def __init__(self):
        self.total_cost = 0.0
        self.total_latency_ms = 0

    def _ret(self, name: str, data: Dict[str, Any], cost: float = 0.02, latency_ms: int = 30) -> Dict[str, Any]:
        self.total_cost += cost
        self.total_latency_ms += latency_ms
        return {"tool": name, "ok": True, "cost": cost, "latency_ms": latency_ms, "data": data}

    def check_weather(self) -> Dict[str, Any]:
        temp = max(-5, min(15, 5 + int(np.random.randn() * 3)))
        return self._ret("check_weather", {"temp_c": temp, "status": "cold" if temp < 8 else "mild"})

    def find_shelter(self) -> Dict[str, Any]:
        return self._ret("find_shelter", {"location": "cave_ne_200m", "capacity": 2})

    def prepare_warmth_kit(self) -> Dict[str, Any]:
        return self._ret("prepare_warmth_kit", {"items": ["blanket", "firestarter", "insulation"]})

    def scan_inventory(self) -> Dict[str, Any]:
        items = ["berries", "sticks", "stone_knife", "cordage"]
        return self._ret("scan_inventory", {"items": items})

    def forage(self) -> Dict[str, Any]:
        found = ["berries", "mushrooms"] if np.random.rand() > 0.3 else ["berries"]
        return self._ret("forage", {"found": found})

    def collect_wood(self) -> Dict[str, Any]:
        qty = 3 + int(np.random.rand() * 4)
        return self._ret("collect_wood", {"bundles": qty})

    def ignite_fire(self) -> Dict[str, Any]:
        return self._ret("ignite_fire", {"status": "lit", "duration_min": 45})

    def inspect_shelter(self) -> Dict[str, Any]:
        return self._ret("inspect_shelter", {"status": "stable", "leaks": 1})

    def reinforce_shelter(self) -> Dict[str, Any]:
        return self._ret("reinforce_shelter", {"status": "reinforced", "materials": ["wood", "cordage"]})

    def craft_tool(self) -> Dict[str, Any]:
        return self._ret("craft_tool", {"made": "stone_axe"})

    def replan_resources(self) -> Dict[str, Any]:
        return self._ret("replan_resources", {"status": "rebalanced", "focus": ["warmth", "food"]})
