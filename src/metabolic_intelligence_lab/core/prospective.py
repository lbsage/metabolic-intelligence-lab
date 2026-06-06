from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from .memory import GeometricMemory


class ProspectiveEngine:
    """Predictive probe over memory; surfaces likely-near-future concepts."""

    def __init__(self, memory: GeometricMemory):
        self.memory = memory

    def simulate(self, current_vec: np.ndarray, threshold: float = 0.65, topk: int = 3) -> List[str]:
        forward = self.memory.retrieve(current_vec, threshold=threshold)
        return [label for _, label in forward[:topk]]


class ProspectiveLog:
    """Log simulations, outcomes, and hit/miss learning signals."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def record(self, t: int, context_id: str, forecast: List[str]) -> int:
        eid = len(self.entries)
        self.entries.append({
            "id": eid,
            "t": t,
            "context": context_id,
            "forecast": list(forecast),
            "observed": [],
            "score": None,
        })
        return eid

    def observe(self, eid: int, observed: List[str]) -> float:
        if 0 <= eid < len(self.entries):
            self.entries[eid]["observed"] = list(observed)
            score = self._score(self.entries[eid]["forecast"], observed)
            self.entries[eid]["score"] = score
            return score
        return 0.0

    @staticmethod
    def _score(forecast: List[str], observed: List[str]) -> float:
        if not forecast and not observed:
            return 1.0
        if not forecast or not observed:
            return 0.0
        inter = len(set(forecast) & set(observed))
        union = len(set(forecast) | set(observed))
        return inter / max(1, union)

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": self.entries}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProspectiveLog":
        pl = ProspectiveLog()
        pl.entries = list(d.get("entries", []))
        return pl
