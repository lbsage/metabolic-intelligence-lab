from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from .memory import GeometricMemory


class ProspectiveEngine:
    """Predictive probe over memory; surfaces likely-near-future concepts."""

    def __init__(self, memory: GeometricMemory):
        self.memory = memory

    def simulate(self, current_vec: np.ndarray, threshold: float = 0.65, topk: int = 3) -> List[str]:
        forward = self.memory.retrieve(current_vec, threshold=threshold)
        return [label for _, label in forward[:topk]]

    def simulate_horizon(
        self,
        current_vec: np.ndarray,
        depth: int = 3,
        threshold: float = 0.65,
        topk: int = 3,
        drift: float = 0.35,
    ) -> List[List[str]]:
        """Multi-step look-ahead forecast.

        Chains single-step retrievals into a trajectory: at each step the probe
        vector is drifted toward the best-matching memory vector before the next
        retrieval, projecting how the context is likely to evolve rather than
        only what it currently resembles.
        """
        label_vectors: Dict[str, np.ndarray] = {}
        for m in self.memory.items:
            label_vectors.setdefault(m["label"], m["vector"])

        trajectory: List[List[str]] = []
        probe = np.array(current_vec, dtype=float)
        for _ in range(max(1, depth)):
            matches = self.memory.retrieve(probe, threshold=threshold)
            labels = [label for _, label in matches[:topk]]
            trajectory.append(labels)
            if not labels:
                break
            target = label_vectors.get(labels[0])
            if target is not None:
                probe = (1.0 - drift) * probe + drift * np.array(target, dtype=float)
        return trajectory

    @staticmethod
    def flatten_horizon(trajectory: List[List[str]]) -> List[str]:
        """Union of forecast labels across all steps, ordered by first appearance."""
        seen: List[str] = []
        for step_labels in trajectory:
            for label in step_labels:
                if label not in seen:
                    seen.append(label)
        return seen


class ProspectiveLog:
    """Log simulations, outcomes, and hit/miss learning signals."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def record(self, t: int, context_id: str, forecast: List[str], trajectory: Optional[List[List[str]]] = None) -> int:
        eid = len(self.entries)
        entry = {
            "id": eid,
            "t": t,
            "context": context_id,
            "forecast": list(forecast),
            "observed": [],
            "score": None,
        }
        if trajectory is not None:
            entry["trajectory"] = [list(step) for step in trajectory]
        self.entries.append(entry)
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

    def accuracy_summary(self) -> Dict[str, Any]:
        """Aggregate hit-rate stats over all scored (observed) entries."""
        scored = [e["score"] for e in self.entries if e.get("score") is not None]
        if not scored:
            return {"n_scored": 0, "avg_score": 0.0, "hit_rate": 0.0}
        hits = sum(1 for s in scored if s > 0.0)
        return {
            "n_scored": len(scored),
            "avg_score": float(sum(scored) / len(scored)),
            "hit_rate": float(hits / len(scored)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": self.entries}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProspectiveLog":
        pl = ProspectiveLog()
        pl.entries = list(d.get("entries", []))
        return pl
