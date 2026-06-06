from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from metabolic_intelligence_lab.core.agent import DUSEAgent
from metabolic_intelligence_lab.core.ternary_field import TernaryField


class LiveViz:
    """Live plot of top salience vs threshold + ternary coupling heatmap."""

    def __init__(self, labels: List[str]):
        self.labels = labels
        self.fig, (self.ax_line, self.ax_heat) = plt.subplots(1, 2, figsize=(11, 4))

    def _build_coupling_matrix(self, tf: TernaryField) -> np.ndarray:
        n = len(self.labels)
        m = np.zeros((n, n), dtype=float)
        idx = {lb: i for i, lb in enumerate(self.labels)}
        for (a, b), w in tf.coupling.items():
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                m[i, j] = m[j, i] = w
        return m

    def update(self, t: int, agent: DUSEAgent) -> None:
        self.ax_line.cla()
        self.ax_line.set_title("Top-Salience vs Threshold")
        self.ax_line.set_xlabel("t")
        self.ax_line.set_ylabel("value")
        self.ax_line.plot(agent._viz_salience_hist, label="top_salience")
        self.ax_line.plot(agent._viz_threshold_hist, label="threshold")
        self.ax_line.legend(loc="upper right")

        self.ax_heat.cla()
        self.ax_heat.set_title("Ternary Coupling Heatmap")
        if agent.ternary_field:
            m = self._build_coupling_matrix(agent.ternary_field)
            im = self.ax_heat.imshow(m, interpolation="nearest", aspect="auto")
            self.ax_heat.set_xticks(range(len(self.labels)))
            self.ax_heat.set_yticks(range(len(self.labels)))
            self.ax_heat.set_xticklabels(self.labels, rotation=45, ha="right")
            self.ax_heat.set_yticklabels(self.labels)
        plt.tight_layout()
        plt.pause(0.001)
