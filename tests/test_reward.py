from metabolic_intelligence_lab.core.reward import RewardModel
from metabolic_intelligence_lab.core.world import WorldState

def test_reward_eval():
    r = RewardModel()
    metrics = r.eval(WorldState(), energy_used=0.1, tool_cost=0.1, latency_ms=30, depth=1)
    assert "ces_like" in metrics
