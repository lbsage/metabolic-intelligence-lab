"""
Deterministic replay from a saved experiment directory.

Loads the config.yaml from a previous run, re-seeds the RNG, re-runs via
System0Sandbox, then prints a side-by-side comparison of the original and
replay summary metrics.

Usage:
    python experiments/replay_experiment.py results/baseline_demo/<run_id>
"""
import argparse
import json
from pathlib import Path

from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.system0_sandbox import System0Sandbox


def replay(run_dir: str | Path, verbose: bool = False) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")

    cfg = ExperimentConfig.from_yaml(config_path)
    print(f"Replaying: {cfg.name}  seed={cfg.seed}  steps={cfg.steps}")

    # Original summary
    original_summary_path = run_dir / "summary.json"
    original_summary = {}
    if original_summary_path.exists():
        with original_summary_path.open() as f:
            original_summary = json.load(f)

    # Re-run with identical config (same seed → deterministic)
    sb = System0Sandbox(cfg, name="replay")
    sb.run(verbose=verbose)
    replay_summary = sb.summary

    # Compare
    print("\n--- Original vs Replay ---")
    fields = ["avg_reward", "avg_ces_like", "avg_energy", "n_plans", "survival_rate"]
    header = f"{'metric':<22} {'original':>12} {'replay':>12}"
    print(header)
    print("-" * len(header))
    for field in fields:
        orig_val = original_summary.get(field, "n/a")
        repl_val = replay_summary.get(field, "n/a")
        orig_str = f"{orig_val:.4f}" if isinstance(orig_val, float) else str(orig_val)
        repl_str = f"{repl_val:.4f}" if isinstance(repl_val, float) else str(repl_val)
        print(f"{field:<22} {orig_str:>12} {repl_str:>12}")

    # Save replay run
    replay_dir = run_dir.parent / f"{run_dir.name}_replay"
    sb.save(replay_dir)
    print(f"\nReplay saved to {replay_dir}")
    return {"original": original_summary, "replay": replay_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved experiment run.")
    parser.add_argument("run_dir", help="Path to a saved results/<name>/<run_id> directory")
    parser.add_argument("--verbose", action="store_true", help="Print per-step output")
    args = parser.parse_args()
    replay(args.run_dir, verbose=args.verbose)
