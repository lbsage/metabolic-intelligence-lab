# Metabolic Intelligence Lab

A research sandbox for **System 0 / DUSE micro-agents**: energy-aware cognition, salience regulation, geometric memory, prospective memory, ternary field dynamics, goal utility frontiers, budgeted tool use, and SDF/CES-inspired reward modeling.

## What is here

This repo packages the System 0 Cognition Sandbox into a reproducible research layout.

Core concepts implemented:

- Geometric memory with decay and reinforcement
- Salience engine with novelty/frequency weighting
- Prospective memory log
- Ternary salience field with coupling dynamics
- Energy budget and adaptive policy module
- DUSE micro-agent wrapper
- Tiny world state
- Preconditions/effects planner with branching/replanning
- Budgeted mock tool caller
- Multi-goal utility frontier
- SDF/CES-inspired reward/cost model
- YAML experiment configs
- Versioned experiment snapshots and telemetry
- Comparative experiment harness

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

python experiments/run_demo.py
python experiments/run_compare.py
python experiments/run_sweep.py
```

Outputs are written to:

```text
results/<experiment_name>/<run_id>/
├── config.yaml
├── summary.json
├── agent_snapshot.json
├── telemetry.json
└── telemetry.csv
```

## Run a specific config

```bash
python experiments/run_compare.py --config configs/cold_budget_high.yaml --steps 50
```

## Current experimental frame

The sandbox compares different “cognitive phenotypes” by changing:

- salience threshold
- energy reserve
- tool budget
- cost/time penalties
- coupling reinforcement/decay
- memory decay
- prospective depth

The default comparative harness contrasts:

- `agent_energy`: more cost-sensitive, conservative
- `agent_reward`: more value-seeking, willing to spend more

## Design thesis

This project explores whether intelligence can be modeled as:

> energy-constrained salience regulation over prospective memory and compositional action.

That means the agent does not “think harder” by default. It uses salience and metabolic constraints to decide when deeper reasoning, planning, memory replay, or tool use is worth the cost.
