from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Tuple


class SalienceEngine:
    """Temporal + novelty-weighted salience with decay."""

    def __init__(self, maxlen: int = 256):
        self.trace: deque[str] = deque(maxlen=maxlen)
        self.weights: Dict[str, float] = {}

    def observe(self, label: str, novelty: float = 1.0) -> None:
        self.trace.append(label)
        incr = novelty * (1.0 / (1.0 + self.trace.count(label)))
        self.weights[label] = self.weights.get(label, 0.0) + incr

    def decay(self, rate: float = 0.01) -> None:
        for k in list(self.weights.keys()):
            self.weights[k] *= (1.0 - rate)
            if self.weights[k] < 0.01:
                del self.weights[k]

    def top(self, k: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {"trace": list(self.trace), "weights": self.weights}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SalienceEngine":
        se = SalienceEngine(maxlen=max(16, len(d.get("trace", [])) or 256))
        for x in d.get("trace", []):
            se.trace.append(x)
        se.weights = dict(d.get("weights", {}))
        return se
