from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.harness import run_single_config

if __name__ == "__main__":
    cfg = ExperimentConfig(name="baseline_demo", steps=20, visualize=False)
    out = run_single_config(cfg)
    print(f"Saved run to {out}")
