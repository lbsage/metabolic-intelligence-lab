"""Single-agent demo using System0Sandbox — the canonical entry point."""
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.system0_sandbox import System0Sandbox

if __name__ == "__main__":
    cfg = ExperimentConfig(name="baseline_demo", steps=20, visualize=False)
    sb = System0Sandbox(cfg)
    sb.run()
    out = sb.save(f"results/{cfg.name}/{System0Sandbox.run_id()}")
    print(f"\nSaved run to {out}")
    import json
    print(json.dumps(sb.summary, indent=2))
