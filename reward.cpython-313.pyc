from __future__ import annotations
from typing import Any, Dict


class WorldState:
    """Minimal world model the planner can read/write."""

    def __init__(self):
        self.weather: Dict[str, Any] = {"temp_c": 10, "status": "mild"}
        self.inventory: Dict[str, int] = {"berries": 1, "sticks": 0, "cordage": 0, "stone_knife": 1}
        self.shelter: Dict[str, Any] = {"status": "basic", "leaks": 2}
        self.fire: Dict[str, Any] = {"lit": False, "duration_min": 0}
        self.hunger: int = 0

    def has(self, item: str, qty: int = 1) -> bool:
        return self.inventory.get(item, 0) >= qty

    def add(self, item: str, qty: int = 1) -> None:
        self.inventory[item] = self.inventory.get(item, 0) + qty

    def consume(self, item: str, qty: int = 1) -> bool:
        if self.has(item, qty):
            self.inventory[item] -= qty
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weather": self.weather,
            "inventory": self.inventory,
            "shelter": self.shelter,
            "fire": self.fire,
            "hunger": self.hunger,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorldState":
        w = WorldState()
        w.weather = dict(d.get("weather", w.weather))
        w.inventory = dict(d.get("inventory", w.inventory))
        w.shelter = dict(d.get("shelter", w.shelter))
        w.fire = dict(d.get("fire", w.fire))
        w.hunger = int(d.get("hunger", 0))
        return w
