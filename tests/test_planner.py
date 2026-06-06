from metabolic_intelligence_lab.core.planner import RulePlanner
from metabolic_intelligence_lab.core.tools import MockToolAPI
from metabolic_intelligence_lab.core.world import WorldState
from metabolic_intelligence_lab.core.telemetry import Telemetry

def test_planner_constructs():
    p = RulePlanner(MockToolAPI(), WorldState(), Telemetry())
    assert "cold" in p.plan_map
