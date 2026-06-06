# metabolic-intelligence-lab

A research framework for energy-aware cognition, salience-driven intelligence, and adaptive cognitive architectures.

**metabolic-intelligence-lab** explores an alternative path toward intelligent systems: treating cognition not as pure next-token prediction, but as a dynamic process of salience regulation, prospective simulation, resource allocation, field coupling, and adaptive constraint management.

The project combines ideas from:

- active inference & the Free Energy Principle (FEP)
- embodied and prospective cognition
- geometric and field-based memory models
- metabolic and energy-aware computation
- compositional and modular intelligence architectures
- experimental AI ecology

At its core is the **System 0 sandbox** — a configurable experimental environment for studying:

- salience-driven behavior
- multi-goal arbitration
- prospective memory
- energy-constrained planning
- ternary field dynamics
- CES/SDF-inspired reward systems
- adaptive cognitive phenotypes

Unlike conventional agent frameworks centered primarily on LLM orchestration, this repo treats **energy**, **temporal depth**, **memory geometry**, **field coupling**, and **constraint fluidity** as first-class primitives of intelligence.

The framework supports:

- comparative cognitive experiments
- configurable metabolic policies
- parameter sweeps
- telemetry and replay
- evolutionary search
- future DUSE-based distributed cognition research

---

## Core Themes

| Theme | Description |
|---|---|
| **System 0 cognition** | salience before reasoning; environment as substrate; cognition as adaptive field dynamics |
| **Energy-aware intelligence** | intelligence under constraint; metabolic allocation; efficiency vs depth tradeoffs |
| **Prospective cognition** | simulation before action; multi-temporal memory; embodied rehearsal loops |
| **Constraint fluidity** | adaptive thresholds; dynamic reserve allocation; shifting cognitive regimes |
| **Composable cognitive architectures** | modular agents; DUSE-inspired cognition; distributed salience fields |

---

## What Is Here

### Core Modules

| Module | Role |
|---|---|
| `system0_sandbox.py` | Top-level agent loop orchestrating System 0 / System 2 transitions |
| `configs.py` | Mock tool API — 11 budgeted tools with cost/latency accounting |
| `harness.py` | Task queue with temporal + salience-conditioned prospective memory |
| `policy.py` | FEP-inspired adaptive policy; manages energy reserve and escalation gating |
| `bus.py` | Lightweight message bus for inter-module event routing |
| `world.py` | World state belief model (weather, inventory, shelter, fire, hunger) |
| `telemetry.py` | Step-level telemetry capture, CSV + JSON export |
| `results.py` | Versioned experiment snapshots and SDF/CES reward model |
| `live_viz.py` | Real-time terminal visualization of salience, energy, and field state |
| `replay_experiment.py` | Load and replay saved experiment snapshots |

### Cognitive Primitives (implemented across `__init__` variants)

| Primitive | Description |
|---|---|
| **Geometric Memory** | Vector-based associative memory with exponential decay, cosine similarity retrieval, and reinforcement |
| **Salience Engine** | Novelty/frequency-weighted label tracking with decay over time |
| **Ternary Field** | Coupled `{-1, 0, +1}` dynamics that modulate salience thresholds via pairwise coupling |
| **Prospective Memory** | Temporal task queue with salience-conditional triggering and look-ahead simulation |
| **Rule Planner** | Preconditions/effects planner with branching replanning and budget enforcement |
| **Goal Utility Frontier** | Multi-goal trade-off surface separating goal modeling from execution |
| **SDF/CES Reward Model** | Survival + comfort value minus cost/latency penalties, with CES-like efficiency metric |

### Experiment Configs (`*.yaml`)

| Config | Phenotype |
|---|---|
| `exploratory.yaml` | Low salience threshold, high tool budget, aggressive field coupling |
| `energy_conservative.yaml` | High salience threshold, minimal tool budget, conservative decay |
| `high_depth.yaml` | Moderate threshold, deep prospective simulation (depth=3) |
| `cold_budget_high.yaml` | Temperature-stressed scenario with elevated energy budget |

### Markdown Documentation

| File | Contents |
|---|---|
| `experiment_design.md` | Experimental methodology and comparative harness design |
| `geometric_memory.md` | Phase-by-phase development history and memory architecture |
| `metrics.md` | Output metric definitions (reward, CES efficiency, entropy, stability) |
| `parameter_sweeps.md` | Sweep strategies and phenotype comparison protocol |
| `prospective_memory.md` | Prospective simulation loop and temporal memory design |
| `sdf_ces_framework.md` | SDF/CES-inspired reward and cost modeling rationale |
| `ternary_field.md` | Ternary field coupling dynamics and salience modulation |
| `thinking_with_20_watts.md` | Theoretical grounding: intelligence under human-brain-scale metabolic budget |

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install -e .

python run_demo.py
python run_compare.py
python run_sweep.py
```

### Run a specific config

```bash
python run_compare.py --config cold_budget_high.yaml --steps 50
```

### Output structure

```
results/<experiment_name>/<run_id>/
├── config.yaml
├── summary.json
├── agent_snapshot.json
├── telemetry.json
└── telemetry.csv
```

---

## Architecture

### System 0 / System 2 Loop

```
Sense Context
    │
    ▼
Geometric Memory Retrieval  ──►  Ternary Field Modulation
    │                                      │
    ▼                                      ▼
Salience Update  ◄──────────── Coupled Threshold Adjustment
    │
    ▼
Goal Utility Frontier  ──►  Escalation Decision
                                    │
                         utility > threshold?
                         salience threshold met?
                         energy reserve available?
                                    │
                              YES   │   NO
                               ▼         ▼
                         System 2    Stay in
                         (RulePlanner)  System 0
                               │
                         Budgeted Tool Execution
                               │
                         Outcome vs Forecast
                               │
                         Memory Reinforcement / Decay
```

### Key Design Principles

- **Metabolic gating** — System 2 planning is only triggered when salience, utility, and energy reserve jointly justify the cost.
- **Serialization-first** — Full agent state is snapshotted at every step, enabling deterministic replay and post-hoc analysis.
- **Config-driven phenotypes** — All cognitive parameters are externalized to YAML; no hardcoded behavior.
- **Budget-aware execution** — Every tool call logs its cost and latency against a per-episode budget ceiling.
- **Declarative goal hierarchy** — Multi-goal utility frontier decouples what the agent wants from how it plans.

---

## Comparative Experiment Harness

The default harness contrasts two cognitive phenotypes:

| Phenotype | Behavior |
|---|---|
| `agent_energy` | Cost-sensitive, conservative; high salience threshold, low tool budget |
| `agent_reward` | Value-seeking, willing to spend; lower threshold, higher budget |

Key output metrics:

- `avg_reward` — mean total reward per episode
- `ces_efficiency` — value achieved per unit cost (CES-like)
- `avg_energy` — mean tool cost consumed
- `plan_count` — number of System 2 escalations
- `salience_entropy` — distribution of salience across labels
- `frontier_variance` — spread of goal utility estimates
- `survival_rate` — fraction of steps with positive survival score

---

## Long-Term Vision

The long-term goal is to develop **programmable cognitive ecologies** capable of adaptive, efficient, and scalable intelligence under real-world constraints.

This includes exploring:

- salience engines
- geometric memory substrates
- field-based cognition
- prospective simulation systems
- metabolic intelligence
- new foundations for AGI, ASI, and artificial life

---

## Design Thesis

> Intelligence can be modeled as energy-constrained salience regulation over prospective memory and compositional action.

The agent does not "think harder" by default. It uses salience and metabolic constraints to decide when deeper reasoning, planning, memory replay, or tool use is worth the cost — targeting the operating envelope of the human brain (~20W).

---

## License

See [LICENSE](LICENSE).
