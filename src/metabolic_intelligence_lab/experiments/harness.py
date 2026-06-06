from __future__ import annotations

import json
from pathlib import Path
from typing import List

from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.results import summarize_rows
from metabolic_intelligence_lab.system0_sandbox import System0Sandbox
from metabolic_intelligence_lab.visualization.live_viz import LiveViz

LABELS = ["food", "shelter", "fire", "cold", "tool"]


def run_single_config(cfg: ExperimentConfig, out_root: str | Path = "results") -> Path:
    out_dir = Path(out_root) / cfg.name / System0Sandbox.run_id()

    lv = LiveViz(LABELS) if cfg.visualize else None

    def _viz(t: int, tick, agent) -> None:  # type: ignore[type-arg]
        if lv:
            lv.update(t, agent)

    sandbox = System0Sandbox(cfg)
    sandbox.run(step_callback=_viz)
    return sandbox.save(out_dir)


def run_experiments(configs: List[ExperimentConfig], out_root: str | Path = "results") -> List[Path]:
    output_dirs: List[Path] = []
    aggregate = []
    for cfg in configs:
        out_dir = run_single_config(cfg, out_root=out_root)
        output_dirs.append(out_dir)
        with (out_dir / "summary.json").open("r", encoding="utf-8") as f:
            aggregate.append(json.load(f))

    metrics_dir = Path(out_root) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"summary_{System0Sandbox.run_id()}.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    return output_dirs


def compare_agents(
    cfg_energy: ExperimentConfig,
    cfg_reward: ExperimentConfig,
    out_root: str | Path = "results",
) -> Path:
    out_dir = Path(out_root) / "comparison_runs" / System0Sandbox.run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_energy.save(out_dir / "agent_energy_config.yaml")
    cfg_reward.save(out_dir / "agent_reward_config.yaml")

    sb_e = System0Sandbox(cfg_energy, name="agent_energy")
    sb_r = System0Sandbox(cfg_reward, name="agent_reward")

    steps = max(cfg_energy.steps, cfg_reward.steps)
    print("\n--- Comparative Experiment: agent_energy vs agent_reward ---\n")
    for t in range(steps):
        te = sb_e.step(t) if t < cfg_energy.steps else None
        tr = sb_r.step(t) if t < cfg_reward.steps else None
        if te and tr:
            print(
                f"[t={t:02d}] E: {te['sense']['top_label'] or '—':<8} "
                f"E={te['energy_used']:.2f} thr={te['threshold']:.2f} | "
                f"R: {tr['sense']['top_label'] or '—':<8} "
                f"E={tr['energy_used']:.2f} thr={tr['threshold']:.2f}"
            )

    sb_e.agent.save_json(out_dir / "agent_energy.json")
    sb_e.agent.log.write_json(out_dir / "agent_energy_telemetry.json")
    sb_e.agent.log.write_csv(out_dir / "agent_energy_telemetry.csv")

    sb_r.agent.save_json(out_dir / "agent_reward.json")
    sb_r.agent.log.write_json(out_dir / "agent_reward_telemetry.json")
    sb_r.agent.log.write_csv(out_dir / "agent_reward_telemetry.csv")

    summary = {
        "agent_energy": summarize_rows("agent_energy", sb_e.agent.log.rows).to_dict(),
        "agent_reward": summarize_rows("agent_reward", sb_r.agent.log.rows).to_dict(),
    }
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Comparison Summary ---")
    print(json.dumps(summary, indent=2))
    return out_dir
