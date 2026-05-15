import argparse
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.harness import compare_agents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    energy = ExperimentConfig(
        name="agent_energy",
        salience_threshold=1.2,
        reserve=0.35,
        gamma_cost=1.0,
        delta_time=0.4,
        tool_budget=0.25,
        steps=args.steps,
        seed=42,
    )
    reward = ExperimentConfig(
        name="agent_reward",
        salience_threshold=1.2,
        reserve=0.25,
        gamma_cost=0.4,
        delta_time=0.2,
        tool_budget=0.45,
        steps=args.steps,
        seed=43,
    )
    out = compare_agents(energy, reward)
    print(f"Saved comparison to {out}")
