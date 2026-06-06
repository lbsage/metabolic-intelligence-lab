import pytest
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig


def test_config_defaults():
    cfg = ExperimentConfig(name="test")
    assert cfg.salience_threshold == 1.2


def test_config_all_defaults():
    cfg = ExperimentConfig(name="test")
    assert cfg.reserve == 0.3
    assert cfg.tool_budget == 0.3
    assert cfg.steps == 30
    assert cfg.seed == 42
    assert cfg.prospective_depth == 1
    assert cfg.visualize is False


def test_config_custom_values():
    cfg = ExperimentConfig(name="x", salience_threshold=1.5, tool_budget=0.5, steps=10)
    assert cfg.salience_threshold == 1.5
    assert cfg.tool_budget == 0.5
    assert cfg.steps == 10


# -- Validation ------------------------------------------------------------

def test_invalid_salience_threshold_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", salience_threshold=3.0)


def test_invalid_reserve_above_095_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", reserve=0.99)


def test_invalid_negative_tool_budget_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", tool_budget=-0.1)


def test_invalid_prospective_depth_zero_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", prospective_depth=0)


def test_invalid_memory_decay_ge1_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(name="bad", memory_decay=1.0)


# -- Serialization ---------------------------------------------------------

def test_to_dict_contains_all_fields():
    cfg = ExperimentConfig(name="test")
    d = cfg.to_dict()
    for field in ("name", "salience_threshold", "reserve", "tool_budget",
                  "steps", "seed", "memory_decay", "coupling_gamma"):
        assert field in d


def test_save_and_from_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig(name="roundtrip", salience_threshold=1.35, steps=15, seed=99)
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    cfg2 = ExperimentConfig.from_yaml(path)
    assert cfg2.name == "roundtrip"
    assert cfg2.salience_threshold == 1.35
    assert cfg2.steps == 15
    assert cfg2.seed == 99


# -- apply_to --------------------------------------------------------------

def test_apply_to_sets_agent_and_planner_params():
    from metabolic_intelligence_lab.core.agent import DUSEAgent
    from metabolic_intelligence_lab.core.planner import RulePlanner
    from metabolic_intelligence_lab.core.telemetry import Telemetry
    from metabolic_intelligence_lab.core.tools import MockToolAPI
    from metabolic_intelligence_lab.core.world import WorldState

    cfg = ExperimentConfig(name="t", salience_threshold=1.4, tool_budget=0.25,
                           memory_decay=0.02, reserve=0.35)
    agent = DUSEAgent(name="a")
    agent.seed(["fire", "food"])
    planner = RulePlanner(MockToolAPI(), WorldState(), Telemetry())
    cfg.apply_to(agent, planner)

    assert agent.salience_threshold == 1.4
    assert agent.gm.memory_decay == 0.02
    assert agent.energy.reserve == 0.35
    assert planner.tool_budget == 0.25
