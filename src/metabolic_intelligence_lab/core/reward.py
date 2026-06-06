from __future__ import annotations
from typing import Dict
import numpy as np
from .world import WorldState


class RewardModel:
    """SDF/CES-inspired evaluator over world state and resource usage."""

    def __init__(
        self,
        alpha_survival: float = 1.0,
        beta_comfort: float = 0.5,
        gamma_cost: float = 0.7,
        delta_time: float = 0.3,
    ):
        self.alpha_survival = alpha_survival
        self.beta_comfort = beta_comfort
        self.gamma_cost = gamma_cost
        self.delta_time = delta_time

    @staticmethod
    def _survival_score(w: WorldState) -> float:
        temp = w.weather.get("temp_c", 10)
        fire_lit = bool(w.fire.get("lit", False))
        leaks = int(w.shelter.get("leaks", 0))
        hunger = int(w.hunger)

        if temp < -5:
            warmth = 0.1
        elif temp < 0:
            warmth = 0.3
        elif temp < 8:
            warmth = 0.6
        else:
            warmth = 0.8
        if fire_lit:
            warmth = min(1.0, warmth + 0.2)

        if leaks >= 3:
            shelter = 0.2
        elif leaks == 2:
            shelter = 0.4
        elif leaks == 1:
            shelter = 0.6
        else:
            shelter = 0.8

        hunger_term = max(0.0, 1.0 - 0.05 * hunger)
        return float(np.clip(0.4 * warmth + 0.4 * shelter + 0.2 * hunger_term, 0.0, 1.0))

    @staticmethod
    def _comfort_score(w: WorldState) -> float:
        temp = w.weather.get("temp_c", 10)
        leaks = int(w.shelter.get("leaks", 0))
        fire_lit = bool(w.fire.get("lit", False))

        if temp < -5:
            base = 0.1
        elif temp < 0:
            base = 0.3
        elif temp < 8:
            base = 0.5
        else:
            base = 0.7
        if fire_lit:
            base += 0.1
        base -= 0.1 * min(leaks, 3)
        return float(np.clip(base, 0.0, 1.0))

    def eval(
        self,
        world: WorldState,
        energy_used: float,
        tool_cost: float,
        latency_ms: int,
        depth: int = 1,
    ) -> Dict[str, float]:
        surv = self._survival_score(world)
        comfort = self._comfort_score(world)
        resource_cost = float(max(0.0, energy_used) + max(0.0, tool_cost))
        time_cost = max(0.0, latency_ms / 1000.0)
        value = self.alpha_survival * surv + self.beta_comfort * comfort
        penalty = self.gamma_cost * resource_cost + self.delta_time * time_cost
        total_reward = value - penalty
        denom = max(1e-6, resource_cost + 0.1 * time_cost)
        ces_like = value / denom
        return {
            "survival": surv,
            "comfort": comfort,
            "resource_cost": resource_cost,
            "time_cost": time_cost,
            "depth": float(depth),
            "value": value,
            "penalty": penalty,
            "total_reward": total_reward,
            "ces_like": ces_like,
        }
