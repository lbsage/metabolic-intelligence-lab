import numpy as np
from metabolic_intelligence_lab.core.memory import GeometricMemory
from metabolic_intelligence_lab.core.prospective import ProspectiveEngine, ProspectiveLog


# -- ProspectiveEngine -----------------------------------------------------

def _engine_with_memory():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "fire", strength=0.6)
    gm.store(np.array([0.0, 1.0]), "food", strength=0.6)
    return ProspectiveEngine(gm)


def test_simulate_returns_list():
    pe = _engine_with_memory()
    result = pe.simulate(np.array([1.0, 0.0]))
    assert isinstance(result, list)


def test_simulate_top_match_is_closest():
    pe = _engine_with_memory()
    result = pe.simulate(np.array([1.0, 0.0]), threshold=0.5, topk=1)
    assert result == ["fire"]


def test_simulate_empty_when_nothing_matches():
    pe = _engine_with_memory()
    result = pe.simulate(np.array([0.5, 0.5]), threshold=0.99, topk=3)
    assert result == []


# -- ProspectiveEngine.simulate_horizon ------------------------------------

def test_simulate_horizon_returns_list_of_lists():
    pe = _engine_with_memory()
    trajectory = pe.simulate_horizon(np.array([1.0, 0.0]), depth=3, threshold=0.5)
    assert isinstance(trajectory, list)
    assert len(trajectory) <= 3
    for step in trajectory:
        assert isinstance(step, list)


def test_simulate_horizon_first_step_matches_simulate():
    pe = _engine_with_memory()
    trajectory = pe.simulate_horizon(np.array([1.0, 0.0]), depth=1, threshold=0.5, topk=1)
    single = pe.simulate(np.array([1.0, 0.0]), threshold=0.5, topk=1)
    assert trajectory[0] == single


def test_simulate_horizon_stops_when_no_match():
    pe = _engine_with_memory()
    trajectory = pe.simulate_horizon(np.array([0.5, 0.5]), depth=3, threshold=0.99)
    assert trajectory == [[]]


def test_flatten_horizon_dedupes_preserving_order():
    pe = _engine_with_memory()
    flat = pe.flatten_horizon([["fire", "food"], ["food", "fire"], ["shelter"]])
    assert flat == ["fire", "food", "shelter"]


# -- ProspectiveLog --------------------------------------------------------

def test_record_returns_int_id():
    pl = ProspectiveLog()
    eid = pl.record(t=0, context_id="agent", forecast=["fire"])
    assert isinstance(eid, int)
    assert eid == 0


def test_record_increments_id():
    pl = ProspectiveLog()
    e0 = pl.record(0, "agent", ["fire"])
    e1 = pl.record(1, "agent", ["food"])
    assert e1 == e0 + 1


def test_observe_perfect_hit():
    pl = ProspectiveLog()
    eid = pl.record(0, "agent", ["fire"])
    score = pl.observe(eid, ["fire"])
    assert score == 1.0


def test_observe_complete_miss():
    pl = ProspectiveLog()
    eid = pl.record(0, "agent", ["fire"])
    score = pl.observe(eid, ["food"])
    assert score == 0.0


def test_observe_partial_overlap():
    pl = ProspectiveLog()
    eid = pl.record(0, "agent", ["fire", "food"])
    score = pl.observe(eid, ["fire", "shelter"])
    # intersection={"fire"}, union={"fire","food","shelter"} → 1/3
    assert abs(score - 1 / 3) < 1e-9


def test_observe_both_empty():
    pl = ProspectiveLog()
    eid = pl.record(0, "agent", [])
    score = pl.observe(eid, [])
    assert score == 1.0


def test_observe_invalid_eid_returns_zero():
    pl = ProspectiveLog()
    assert pl.observe(999, ["fire"]) == 0.0


def test_serialization_roundtrip():
    pl = ProspectiveLog()
    pl.record(0, "agent", ["fire"])
    pl.observe(0, ["fire"])
    d = pl.to_dict()
    pl2 = ProspectiveLog.from_dict(d)
    assert pl2.entries[0]["score"] == 1.0


def test_record_stores_trajectory_when_given():
    pl = ProspectiveLog()
    eid = pl.record(0, "agent", ["fire"], trajectory=[["fire"], ["food"]])
    assert pl.entries[eid]["trajectory"] == [["fire"], ["food"]]


def test_record_omits_trajectory_when_not_given():
    pl = ProspectiveLog()
    eid = pl.record(0, "agent", ["fire"])
    assert "trajectory" not in pl.entries[eid]


def test_accuracy_summary_empty_when_unscored():
    pl = ProspectiveLog()
    pl.record(0, "agent", ["fire"])
    s = pl.accuracy_summary()
    assert s == {"n_scored": 0, "avg_score": 0.0, "hit_rate": 0.0}


def test_accuracy_summary_aggregates_scored_entries():
    pl = ProspectiveLog()
    e0 = pl.record(0, "agent", ["fire"])
    e1 = pl.record(1, "agent", ["food"])
    pl.observe(e0, ["fire"])
    pl.observe(e1, ["shelter"])
    s = pl.accuracy_summary()
    assert s["n_scored"] == 2
    assert s["hit_rate"] == 0.5
    assert abs(s["avg_score"] - 0.5) < 1e-9
