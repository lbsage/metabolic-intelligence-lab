from __future__ import annotations
from typing import Any, Dict, List
import csv
import json
from pathlib import Path


class Telemetry:
    """Simple CSV/JSON logger for agent and planner activity."""

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def log(self, t: int, agent_name: str, tag: str, extra: Dict[str, Any] | None = None) -> None:
        row = {"t": t, "agent": agent_name, "tag": tag}
        if extra:
            row.update(extra)
        self.rows.append(row)

    def write_csv(self, path: str | Path) -> None:
        if not self.rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in self.rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2, default=str)
