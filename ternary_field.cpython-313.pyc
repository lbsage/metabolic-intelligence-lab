from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np


class TernaryField:
    """A minimal ternary (-1, 0, +1) field modulating salience via couplings."""

    def __init__(self, labels: List[str], coupling_gamma: float = 0.02, coupling_decay: float = 0.01):
        self.states: Dict[str, int] = {lb: 0 for lb in labels}
        self.coupling_gamma = coupling_gamma
        self.coupling_decay = coupling_decay
        self.coupling: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i < j:
                    self.coupling[(a, b)] = 0.1 * (1 if np.random.rand() > 0.5 else -1)

    def set_coupling_dynamics(self, gamma: float, decay: float) -> None:
        self.coupling_gamma = gamma
        self.coupling_decay = decay

    def step(self, inputs: Dict[str, float], hysteresis: float = 0.15) -> None:
        raw: Dict[str, float] = {k: inputs.get(k, 0.0) for k in self.states}
        for (a, b), w in self.coupling.items():
            raw[a] += w * self.states[b]
            raw[b] += w * self.states[a]
        for k, v in raw.items():
            if v > hysteresis:
                self.states[k] = +1
            elif v < -hysteresis:
                self.states[k] = -1
            else:
                self.states[k] = 0

    def modulate(self, salience: Dict[str, float], gain: float = 0.1) -> Dict[str, float]:
        out = dict(salience)
        for k, s in self.states.items():
            out[k] = max(0.0, salience.get(k, 0.0) + gain * float(s))
        return out

    def reinforce_couplings(self, labels_hit: List[str], gamma: float | None = None, decay: float | None = None) -> None:
        g = self.coupling_gamma if gamma is None else gamma
        d = self.coupling_decay if decay is None else decay
        lh = set(labels_hit)
        for (a, b), w in list(self.coupling.items()):
            if a in lh and b in lh:
                self.coupling[(a, b)] = w * (1.0 + g)
            else:
                self.coupling[(a, b)] = w * (1.0 - d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "states": self.states,
            "coupling_gamma": self.coupling_gamma,
            "coupling_decay": self.coupling_decay,
            "coupling": {f"{a}|{b}": w for (a, b), w in self.coupling.items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TernaryField":
        labels = list(d.get("states", {}).keys())
        tf = TernaryField(labels, coupling_gamma=float(d.get("coupling_gamma", 0.02)), coupling_decay=float(d.get("coupling_decay", 0.01)))
        tf.states = {k: int(v) for k, v in d.get("states", {}).items()}
        tf.coupling = {}
        for k, w in d.get("coupling", {}).items():
            a, b = k.split("|")
            tf.coupling[(a, b)] = float(w)
        return tf
