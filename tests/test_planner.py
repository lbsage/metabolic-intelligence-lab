import pytest
from metabolic_intelligence_lab.core.planner import RulePlanner
from metabolic_intelligence_lab.core.telemetry import Telemetry
from metabolic_intelligence_lab.core.tools import MockToolAPI
from metabolic_intelligence_lab.core.world import WorldState


def _planner(budget: float = 1.0) -> RulePlanner:
    return RulePlanner(MockToolAPI(), WorldState(), Telemetry(), tool_budget=budget)


def test_planner_constructs():
    p = _planner()
    assert "cold" in p.plan_map


def test_planner_all_goals_in_plan_map():
    p = _planner()
    for goal in ("cold", "food", "fire", "shelter", "tool"):
        assert goal in p.plan_map


# -- Preconditions ---------------------------------------------------------

def test_pre_ok_ignite_fire_fails_without_sticks():
    p = _planner()
    p.world.inventory["sticks"] = 0
    assert not p._pre_ok("ignite_fire")


def test_pre_ok_ignite_fire_passes_with_sticks():
    p = _planner()
    p.world.inventory["sticks"] = 3
    assert p._pre_ok("ignite_fire")


def test_pre_ok_reinforce_shelter_fails_without_sticks():
    p = _planner()
    p.world.inventory["sticks"] = 0
    assert not p._pre_ok("reinforce_shelter")


def test_pre_ok_reinforce_shelter_passes_with_sticks():
    p = _planner()
    p.world.inventory["sticks"] = 2
    assert p._pre_ok("reinforce_shelter")


def test_pre_ok_other_actions_always_true():
    p = _planner()
    for act in ("forage", "scan_inventory", "find_shelter", "check_weather"):
        assert p._pre_ok(act)


# -- Branching regression: shelter must not loop --------------------------

def test_shelter_plan_terminates_without_cordage():
    """Regression: reinforce_shelter used to require cordage, which no tool
    provides, causing an infinite branch loop until budget exhaustion."""
    p = _planner(budget=2.0)
    p.world.inventory["sticks"] = 0   # will need collect_wood first
    p.world.inventory["cordage"] = 0  # cordage unavailable — must not loop
    # Should complete without hanging; if it loops pytest will time out
    p(
        "scout",
        {"obs": {"t": 0}, "goal": "shelter", "sdf_depth": 1,
         "goal_score": 2.0, "top_salience": 2.0,
         "tool_budget": 2.0, "energy_used": 0.0},
    )
    # Branch should resolve: collect_wood → reinforce_shelter (2 steps)
    tool_log = [r for r in p.log.rows if str(r.get("tag", "")).startswith("plan:shelter")]
    actions = [r["tag"].split(":")[-1] for r in tool_log]
    assert "collect_wood" in actions
    assert "reinforce_shelter" in actions


# -- Budget enforcement ---------------------------------------------------

def test_budget_stops_plan_early():
    p = _planner(budget=0.021)   # just enough for one 0.02 tool call
    p(
        "scout",
        {"obs": {"t": 0}, "goal": "cold", "sdf_depth": 1,
         "goal_score": 2.0, "top_salience": 2.0,
         "tool_budget": 0.021, "energy_used": 0.0},
    )
    tool_log = [r for r in p.log.rows if str(r.get("tag", "")).startswith("plan:cold:")]
    # cold plan = 5 steps; only 1 should have run
    assert len(tool_log) == 1


# -- Fire plan executes ---------------------------------------------------

def test_fire_plan_lights_fire_when_sticks_available():
    p = _planner(budget=1.0)
    p.world.inventory["sticks"] = 5
    p(
        "scout",
        {"obs": {"t": 0}, "goal": "fire", "sdf_depth": 1,
         "goal_score": 2.0, "top_salience": 2.0,
         "tool_budget": 1.0, "energy_used": 0.0},
    )
    assert p.world.fire.get("lit") is True


def test_fire_plan_branches_to_collect_wood_when_no_sticks():
    p = _planner(budget=1.0)
    p.world.inventory["sticks"] = 0
    p(
        "scout",
        {"obs": {"t": 0}, "goal": "fire", "sdf_depth": 1,
         "goal_score": 2.0, "top_salience": 2.0,
         "tool_budget": 1.0, "energy_used": 0.0},
    )
    # collect_wood should have been called and fire lit
    assert p.world.fire.get("lit") is True
