from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


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
    """
    Lightweight message bus for inter-agent routing.

    Supports:
    - Point-to-point and broadcast (dst="*") delivery
    - Topic-prefix subscriptions: agents receive only messages whose topic
      starts with their subscribed prefix
    - Bounded message history for replay and debugging
    - Routing statistics
    """

    HISTORY_MAXLEN = 512

    def __init__(self):
        self._queues: Dict[str, deque[Message]] = {}
        self._subscriptions: Dict[str, Optional[str]] = {}   # agent_id → topic_prefix or None
        self._history: Deque[Message] = deque(maxlen=self.HISTORY_MAXLEN)
        self._sent_total: int = 0
        self._delivered_total: int = 0
        self._topic_counts: Dict[str, int] = {}

    def register(self, agent_id: str) -> None:
        """Register an agent to receive messages. Call once per agent."""
        self._queues.setdefault(agent_id, deque())
        self._subscriptions.setdefault(agent_id, None)

    def subscribe(self, agent_id: str, topic_prefix: str) -> None:
        """Filter delivered messages to those whose topic starts with ``topic_prefix``."""
        self._subscriptions[agent_id] = topic_prefix

    def unsubscribe(self, agent_id: str) -> None:
        """Remove topic filter; agent receives all messages again."""
        self._subscriptions[agent_id] = None

    def _matches(self, agent_id: str, msg: Message) -> bool:
        prefix = self._subscriptions.get(agent_id)
        return prefix is None or msg.topic.startswith(prefix)

    def send(self, msg: Message) -> None:
        self._sent_total += 1
        self._history.append(msg)
        self._topic_counts[msg.topic] = self._topic_counts.get(msg.topic, 0) + 1

        if msg.dst == "*":
            for aid, q in self._queues.items():
                if self._matches(aid, msg):
                    q.append(msg)
                    self._delivered_total += 1
        elif msg.dst in self._queues:
            if self._matches(msg.dst, msg):
                self._queues[msg.dst].append(msg)
                self._delivered_total += 1

    def recv(self, agent_id: str, max_msgs: int = 8) -> List[Message]:
        q = self._queues.get(agent_id)
        if not q:
            return []
        out: List[Message] = []
        while q and len(out) < max_msgs:
            out.append(q.popleft())
        return sorted(out, key=lambda m: m.priority, reverse=True)

    def history(self, topic_prefix: Optional[str] = None) -> List[Message]:
        """Return recent messages, optionally filtered by topic prefix."""
        if topic_prefix is None:
            return list(self._history)
        return [m for m in self._history if m.topic.startswith(topic_prefix)]

    def stats(self) -> Dict[str, Any]:
        """Return message accounting summary."""
        return {
            "sent_total": self._sent_total,
            "delivered_total": self._delivered_total,
            "registered_agents": list(self._queues.keys()),
            "pending": {aid: len(q) for aid, q in self._queues.items()},
            "topic_counts": dict(self._topic_counts),
        }
