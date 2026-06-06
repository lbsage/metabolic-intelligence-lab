from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import uuid


@dataclass
class Task:
    id: str
    trigger_time: int
    label_if: Optional[str] = None
    min_salience: float = 1.0
    action: str = "noop"


class TaskQueue:
    def __init__(self):
        self.queue: List[Task] = []

    def add(self, trigger_time: int, label_if: Optional[str], min_salience: float, action: str) -> str:
        tid = str(uuid.uuid4())
        self.queue.append(Task(id=tid, trigger_time=trigger_time, label_if=label_if, min_salience=min_salience, action=action))
        return tid

    def due(self, t: int, salience_top: List[Tuple[str, float]]) -> List[Task]:
        top_label = salience_top[0][0] if salience_top else None
        top_val = salience_top[0][1] if salience_top else 0.0
        ready: List[Task] = []
        remain: List[Task] = []
        for task in self.queue:
            cond = (t >= task.trigger_time) and (task.label_if is None or task.label_if == top_label) and (top_val >= task.min_salience)
            (ready if cond else remain).append(task)
        self.queue = remain
        return ready
