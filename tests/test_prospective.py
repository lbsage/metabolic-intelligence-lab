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
