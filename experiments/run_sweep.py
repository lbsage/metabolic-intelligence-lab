from pathlib import Path
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.harness import run_experiments

if __name__ == "__main__":
    configs = [
        ExperimentConfig.from_yaml(Path("configs/baseline.yaml")),
        ExperimentConfig.from_yaml(Path("configs/energy_conservative.yaml")),
        ExperimentConfig.from_yaml(Path("configs/exploratory.yaml")),
        ExperimentConfig.from_yaml(Path("configs/high_depth.yaml")),
        ExperimentConfig.from_yaml(Path("configs/cold_budget_high.yaml")),
    ]
    outs = run_experiments(configs)
    for out in outs:
        print(f"Saved run to {out}")
