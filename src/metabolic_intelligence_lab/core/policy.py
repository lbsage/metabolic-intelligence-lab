from __future__ import annotations
from typing import List, Tuple
import numpy as np


class PolicyModule:
    """Simple precision/energy policy inspired by active inference/FEP ideas."""

    def __init__(self, base_threshold: float = 1.2, min_threshold: float = 0.9, max_threshold: float = 1.6):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.k_threshold = 0.5

    def update(self, last_matches: List[Tuple[float, str]], current_threshold: float, energy_reserve: float) -> Tuple[float, float]:
        if not last_matches:
            surprise = 1.0
        else:
            top = last_matches[0][0]
            surprise = max(0.0, 1.0 - min(top, 1.0))

        new_thresh = np.clip(
            current_threshold - self.k_threshold * (surprise - 0.5),
            self.min_threshold,
            self.max_threshold,
        )
        new_reserve = float(np.clip(energy_reserve + 0.1 * (surprise - 0.5), 0.1, 0.6))
        return float(new_thresh), new_reserve
