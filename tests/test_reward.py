import pytest
from metabolic_intelligence_lab.core.reward import RewardModel
from metabolic_intelligence_lab.core.world import WorldState


def _world(temp_c=10, fire_lit=False, leaks=0, hunger=0):
    w = WorldState()
    w.weather["temp_c"] = temp_c
    w.fire["lit"] = fire_lit
    w.shelter["leaks"] = leaks
    w.hunger = hunger
    return w


def test_reward_eval():
    r = RewardModel()
    metrics = r.eval(WorldState(), energy_used=0.1, tool_cost=0.1, latency_ms=30, depth=1)
    assert "ces_like" in metrics


def test_all_keys_present():
    r = RewardModel()
    metrics = r.eval(_world(), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    for key in ("survival", "comfort", "resource_cost", "time_cost",
                "value", "penalty", "total_reward", "ces_like", "depth"):
        assert key in metrics, f"missing key: {key}"


def test_survival_cold_no_fire_is_low():
    r = RewardModel()
    # leaks=2 → shelter=0.4; temp=-5 → warmth=0.1; no fire → no boost
    # survival = 0.4*0.1 + 0.4*0.4 + 0.2*1.0 = 0.04 + 0.16 + 0.20 = 0.40
    m = r.eval(_world(temp_c=-5, fire_lit=False, leaks=2), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    assert m["survival"] < 0.5


def test_survival_warm_fire_no_leaks_is_high():
    r = RewardModel()
    m = r.eval(_world(temp_c=12, fire_lit=True, leaks=0), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    assert m["survival"] >= 0.7


def test_fire_improves_survival():
    r = RewardModel()
    no_fire = r.eval(_world(temp_c=5, fire_lit=False), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    with_fire = r.eval(_world(temp_c=5, fire_lit=True), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    assert with_fire["survival"] > no_fire["survival"]


def test_leaks_reduce_survival():
    r = RewardModel()
    no_leaks = r.eval(_world(leaks=0), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    many_leaks = r.eval(_world(leaks=3), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    assert many_leaks["survival"] < no_leaks["survival"]


def test_penalty_applied_with_cost():
    r = RewardModel(gamma_cost=1.0, delta_time=0.0)
    m = r.eval(_world(), energy_used=0.5, tool_cost=0.0, latency_ms=0)
    assert m["penalty"] == pytest.approx(0.5, abs=1e-6)


def test_total_reward_decreases_with_cost():
    r = RewardModel()
    cheap = r.eval(_world(), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    expensive = r.eval(_world(), energy_used=1.0, tool_cost=1.0, latency_ms=1000)
    assert expensive["total_reward"] < cheap["total_reward"]


def test_ces_like_positive_when_value_exists():
    r = RewardModel()
    m = r.eval(_world(temp_c=10), energy_used=0.1, tool_cost=0.1, latency_ms=30)
    assert m["ces_like"] > 0.0


def test_ces_like_denom_never_zero():
    r = RewardModel()
    # Zero cost: denom is clamped to 1e-6
    m = r.eval(_world(), energy_used=0.0, tool_cost=0.0, latency_ms=0)
    assert m["ces_like"] > 0.0


def test_depth_stored_in_metrics():
    r = RewardModel()
    m = r.eval(_world(), energy_used=0.0, tool_cost=0.0, latency_ms=0, depth=3)
    assert m["depth"] == 3.0
