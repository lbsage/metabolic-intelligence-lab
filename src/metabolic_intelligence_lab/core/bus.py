from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EnergyBudget:
    total: float = 1.0
    reserve: float = 0.3
    used: float = 0.0

    def can_spend(self, cost: float) -> bool:
        return (self.used + cost) <= (self.total - self.reserve)

    def spend(self, cost: float) -> bool:
        if self.can_spend(cost):
            self.used += cost
            return True
        return False

    def reset(self) -> None:
        self.used = 0.0


@dataclass
class Message:
    src: str
    dst: str
    topic: str
    payload: Dict[str, Any]
    priority: float = 0.5


class DUSEBus:
    def __init__(self):
        self._queues: Dict[str, deque[Message]] = {}

    def register(self, agent_id: str) -> None:
        self._queues.setdefault(agent_id, deque())

    def send(self, msg: Message) -> None:
        if msg.dst == "*":
            for q in self._queues.values():
                q.append(msg)
        elif msg.dst in self._queues:
            self._queues[msg.dst].append(msg)

    def recv(self, agent_id: str, max_msgs: int = 8) -> List[Message]:
        q = self._queues.get(agent_id)
        if not q:
            return []
        out: List[Message] = []
        while q and len(out) < max_msgs:
            out.append(q.popleft())
        return sorted(out, key=lambda m: m.priority, reverse=True)
