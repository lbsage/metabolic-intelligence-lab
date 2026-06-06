from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List
import json
import random
import shutil
import numpy as np

from metabolic_intelligence_lab.core.agent import DUSEAgent
from metabolic_intelligence_lab.core.bus import DUSEBus
from metabolic_intelligence_lab.core.planner import LLMStub, RulePlanner
from metabolic_intelligence_lab.core.tools import MockToolAPI
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.results import summarize_rows
from metabolic_intelligence_lab.visualization.live_viz import LiveViz


LABELS = ["food", "shelter", "fire", "cold", "tool"]


def _run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def build_agent(cfg: ExperimentConfig, name: str, tools: MockToolAPI) -> DUSEAgent:
    agent = DUSEAgent(name=name, salience_threshold=cfg.salience_threshold)
    agent.seed(LABELS)
    planner = RulePlanner(tools, agent.world, agent.log, tool_budget=cfg.tool_budget)
    agent.system2_plugins = [planner, LLMStub()]
    cfg.apply_to(agent, planner)
    return agent


def run_single_config(cfg: ExperimentConfig, out_root: str | Path = "results") -> Path:
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    out_dir = Path(out_root) / cfg.name / _run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")

    bus = DUSEBus()
    tools = MockToolAPI()
    agent = build_agent(cfg, "scout", tools)
    bus.register(agent.id)

    agent.add_prospective_task(now=0, delay=3, label_if="cold", min_salience=1.0, action="prepare_warmth_kit")
    lv = LiveViz(LABELS) if cfg.visualize else None

    for t in range(cfg.steps):
        s = agent.tick(t, np.random.rand(agent.dim), bus)
        _ = bus.recv(agent.id)
        print(f"[{cfg.name} t={t}] E={s['energy_used']:.2f} thr={s['threshold']:.2f} res={s['reserve']:.2f}")
        if lv:
            lv.update(t, agent)

    agent.save_json(out_dir / "agent_snapshot.json")
    agent.log.write_json(out_dir / "telemetry.json")
    agent.log.write_csv(out_dir / "telemetry.csv")

    result = summarize_rows(cfg.name, agent.log.rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    return out_dir


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
    with (metrics_dir / f"summary_{_run_id()}.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    return output_dirs


def compare_agents(cfg_energy: ExperimentConfig, cfg_reward: ExperimentConfig, out_root: str | Path = "results") -> Path:
    seed = min(cfg_energy.seed, cfg_reward.seed)
    np.random.seed(seed)
    random.seed(seed)

    out_dir = Path(out_root) / "comparison_runs" / _run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_energy.save(out_dir / "agent_energy_config.yaml")
    cfg_reward.save(out_dir / "agent_reward_config.yaml")

    bus = DUSEBus()
    tools = MockToolAPI()

    agent_energy = build_agent(cfg_energy, "agent_energy", tools)
    agent_reward = build_agent(cfg_reward, "agent_reward", tools)
    bus.register(agent_energy.id)
    bus.register(agent_reward.id)

    steps = max(cfg_energy.steps, cfg_reward.steps)
    print("\n--- Comparative Experiment: agent_energy vs agent_reward ---\n")
    for t in range(steps):
        ctx = np.random.rand(6)
        s_e = agent_energy.tick(t, ctx, bus)
        s_r = agent_reward.tick(t, ctx * 0.9 + np.random.rand(6) * 0.1, bus)
        _ = bus.recv(agent_energy.id)
        _ = bus.recv(agent_reward.id)
        print(
            f"[t={t}] E.E={s_e['energy_used']:.2f} thr={s_e['threshold']:.2f} res={s_e['reserve']:.2f} "
            f"| R.E={s_r['energy_used']:.2f} thr={s_r['threshold']:.2f} res={s_r['reserve']:.2f}"
        )

    agent_energy.save_json(out_dir / "agent_energy.json")
    agent_energy.log.write_json(out_dir / "agent_energy_telemetry.json")
    agent_energy.log.write_csv(out_dir / "agent_energy_telemetry.csv")

    agent_reward.save_json(out_dir / "agent_reward.json")
    agent_reward.log.write_json(out_dir / "agent_reward_telemetry.json")
    agent_reward.log.write_csv(out_dir / "agent_reward_telemetry.csv")

    summary = {
        "agent_energy": summarize_rows("agent_energy", agent_energy.log.rows).to_dict(),
        "agent_reward": summarize_rows("agent_reward", agent_reward.log.rows).to_dict(),
    }
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Comparison Summary ---")
    print(json.dumps(summary, indent=2))
    return out_dir
