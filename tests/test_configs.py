from metabolic_intelligence_lab.experiments.configs import ExperimentConfig

def test_config_defaults():
    cfg = ExperimentConfig(name="test")
    assert cfg.salience_threshold == 1.2
