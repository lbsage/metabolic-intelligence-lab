# System0 Sandbox Update Log

This file preserves the major changes made during the System0 Sandbox build so we do not recreate the wheel.

## Phase 1
- Geometric memory field
- Salience engine
- Prospective simulation engine
- System 0 cognitive loop

## Phase 2
- Memory visualization
- Memory and salience decay
- System 1/2 salience triggers
- Rust/Mojo porting hooks

## Phase 3
- DUSE micro-agent wrapper
- Energy budget
- In-process message bus
- Agent serialization

## Phase 4
- Prospective memory log
- Ternary salience field simulator
- System-2 plugins
- Persistence
- FEP-inspired policy module

## Phase 5
- Rule-based System-2 planner stub
- Prospective reinforcement
- Temporal prospective tasks
- Live salience/threshold visualization

## Phase 6
- Mock tool API
- Real-ish planner tool execution
- Ternary coupling heatmap

## Phase 7
- Tiny world state
- Preconditions/effects
- Branch/replan planner logic
- CSV/JSON telemetry logger

## Phase 8
- Goal utility for multi-goal arbitration
- Budgeted tool caller coupled to energy budget

## Phase 9
- Multi-goal utility frontier
- SDF/CES-inspired reward/cost model
- Comparative two-agent experiment harness

## Repo Packaging
- Modularized code into core, experiments, visualization, persistence
- Added YAML configs
- Added versioned result snapshots under results/<config>/<run_id>
- Added ExperimentConfig and ExperimentResult
- Added run_demo, run_compare, and run_sweep entrypoints
