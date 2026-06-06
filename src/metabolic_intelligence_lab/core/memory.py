from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np


class GeometricMemory:
    """Geometric memory modeled as vectors in latent space with strength + decay."""

    def __init__(self, dim: int = 6, memory_decay: float = 0.01):
        self.dim = dim
        self.memory_decay = memory_decay
        self._time = 0
        self.items: List[Dict[str, Any]] = []

    def store(self, vector: np.ndarray, label: str, strength: float = 1.0) -> None:
        self.items.append({
            "vector": np.array(vector, dtype=float),
            "timestamp": self._time,
            "label": label,
            "strength": float(strength),
        })
        self._time += 1

    def decay(self, rate: float | None = None) -> None:
        r = self.memory_decay if rate is None else rate
        for m in self.items:
            m["strength"] *= (1.0 - r)

    def retrieve(self, query_vec: np.ndarray, threshold: float = 0.8) -> List[Tuple[float, str]]:
        qv = np.array(query_vec, dtype=float)
        qn = np.linalg.norm(qv) + 1e-9
        out: List[Tuple[float, str]] = []
        for m in self.items:
            vn = np.linalg.norm(m["vector"]) + 1e-9
            sim = float(np.dot(qv, m["vector"]) / (qn * vn))
            wsim = sim * m["strength"]
            if wsim > threshold:
                out.append((wsim, m["label"]))
        return sorted(out, reverse=True)

    def reinforce(self, labels: List[str], alpha: float = 0.05, beta: float = 0.02) -> None:
        hit = set(labels)
        for m in self.items:
            if m["label"] in hit:
                m["strength"] *= (1.0 + alpha)
            else:
                m["strength"] *= (1.0 - beta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "memory_decay": self.memory_decay,
            "items": [
                {
                    "vector": m["vector"].tolist(),
                    "timestamp": m["timestamp"],
                    "label": m["label"],
                    "strength": m["strength"],
                }
                for m in self.items
            ],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GeometricMemory":
        gm = GeometricMemory(dim=int(d.get("dim", 6)), memory_decay=float(d.get("memory_decay", 0.01)))
        for m in d.get("items", []):
            gm.store(np.array(m["vector"], dtype=float), m["label"], strength=float(m.get("strength", 1.0)))
        return gm
