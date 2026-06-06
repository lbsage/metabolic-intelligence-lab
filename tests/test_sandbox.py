import json
import numpy as np
import pytest
from metabolic_intelligence_lab.core.world import WorldState
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.system0_sandbox import (
    System0Sandbox,
    evolve_world,
    world_to_context,
)


# -- world_to_context ------------------------------------------------------

def test_world_to_context_shape():
    w = WorldState()
    v = world_to_context(w, dim=6)
    assert v.shape == (6,)


def test_world_to_context_shape_extended():
    w = WorldState()
    v = world_to_context(w, dim=10)
    assert v.shape == (10,)


def test_world_to_context_values_clipped():
    w = WorldState()
    v = world_to_context(w, dim=6)
    assert np.all(v >= 0.0)
    assert np.all(v <= 1.0)


def test_world_to_context_fire_lit_encodes_1():
    w = WorldState()
    w.fire["lit"] = True
    v = world_to_context(w, dim=6)
    assert v[2] == 1.0


def test_world_to_context_fire_unlit_encodes_0():
    w = WorldState()
    w.fire["lit"] = False
    v = world_to_context(w, dim=6)
    assert v[2] == 0.0


def test_world_to_context_cold_temp_near_zero():
    w = WorldState()
    w.weather["temp_c"] = -5
    v = world_to_context(w, dim=6)
    assert v[0] == pytest.approx(0.0, abs=1e-6)


def test_world_to_context_warm_temp_near_one():
    w = WorldState()
    w.weather["temp_c"] = 15
    v = world_to_context(w, dim=6)
    assert v[0] == pytest.approx(1.0, abs=1e-6)


# -- evolve_world ----------------------------------------------------------

def test_evolve_world_hunger_increments():
    w = WorldState()
    rng = np.random.default_rng(0)
    w.hunger = 0
    evolve_world(w, rng)
    assert w.hunger == 1


def test_evolve_world_hunger_capped_at_20():
    w = WorldState()
    rng = np.random.default_rng(0)
    w.hunger = 20
    evolve_world(w, rng)
    assert w.hunger == 20


def test_evolve_world_fire_decrements_duration():
    w = WorldState()
    rng = np.random.default_rng(0)
    w.fire["lit"] = True
    w.fire["duration_min"] = 15
    evolve_world(w, rng)
    assert w.fire["duration_min"] == 10


def test_evolve_world_fire_extinguishes():
    w = WorldState()
    rng = np.random.default_rng(0)
    w.fire["lit"] = True
    w.fire["duration_min"] = 5
    events = evolve_world(w, rng)
    assert w.fire["lit"] is False
    assert "fire_out" in events


def test_evolve_world_fire_out_event_returned():
    w = WorldState()
    rng = np.random.default_rng(0)
    w.fire["lit"] = True
    w.fire["duration_min"] = 3
    events = evolve_world(w, rng)
    assert events.get("fire_out") is True


def test_evolve_world_hunger_warning_at_10():
    w = WorldState()
    rng = np.random.default_rng(0)
    w.hunger = 9
    events = evolve_world(w, rng)
    assert "hunger_warning" in events


def test_evolve_world_weather_temp_stays_bounded():
    w = WorldState()
    rng = np.random.default_rng(0)
    for _ in range(50):
        evolve_world(w, rng)
    temp = w.weather["temp_c"]
    assert -5 <= temp <= 15


# -- System0Sandbox --------------------------------------------------------

def _cfg(**kw):
    kw.setdefault("steps", 5)
    return ExperimentConfig(name="test", seed=42, **kw)


def test_sandbox_runs_without_error():
    sb = System0Sandbox(_cfg())
    trace = sb.run(verbose=False)
    assert len(trace) == 5


def test_sandbox_trace_has_required_keys():
    sb = System0Sandbox(_cfg())
    trace = sb.run(verbose=False)
    for tick in trace:
        assert "t" in tick
        assert "env" in tick
        assert "sense" in tick
        assert "escalated" in tick
        assert "energy_used" in tick
        assert "world" in tick


def test_sandbox_energy_resets_each_tick():
    sb = System0Sandbox(_cfg())
    trace = sb.run(verbose=False)
    # Energy should never accumulate past total budget across ticks
    for tick in trace:
        assert tick["energy_used"] <= 1.0


def test_sandbox_t_increments():
    sb = System0Sandbox(_cfg())
    trace = sb.run(verbose=False)
    assert [tick["t"] for tick in trace] == list(range(5))


def test_sandbox_save_creates_files(tmp_path):
    sb = System0Sandbox(_cfg())
    sb.run(verbose=False)
    out = sb.save(tmp_path / "run")
    assert (out / "config.yaml").exists()
    assert (out / "agent_snapshot.json").exists()
    assert (out / "telemetry.json").exists()
    assert (out / "telemetry.csv").exists()
    assert (out / "trace.json").exists()
    assert (out / "summary.json").exists()


def test_sandbox_summary_has_expected_keys():
    sb = System0Sandbox(_cfg())
    sb.run(verbose=False)
    s = sb.summary
    for key in ("avg_reward", "avg_ces_like", "avg_energy", "n_plans", "survival_rate"):
        assert key in s, f"missing key: {key}"


def test_sandbox_avg_energy_nonzero_when_plans_run():
    sb = System0Sandbox(_cfg(steps=15, salience_threshold=1.0))
    sb.run(verbose=False)
    assert sb.summary["n_plans"] > 0


def test_sandbox_deterministic_with_same_seed():
    cfg = _cfg(steps=10)
    trace_a = System0Sandbox(cfg).run(verbose=False)
    trace_b = System0Sandbox(cfg).run(verbose=False)
    for a, b in zip(trace_a, trace_b):
        assert a["energy_used"] == b["energy_used"]
        assert a["threshold"] == b["threshold"]


def test_sandbox_step_callback_called():
    calls = []
    sb = System0Sandbox(_cfg())
    sb.run(verbose=False, step_callback=lambda t, tick, agent: calls.append(t))
    assert calls == list(range(5))
