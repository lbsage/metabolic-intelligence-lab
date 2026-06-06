from metabolic_intelligence_lab.core.salience import SalienceEngine


def test_salience_top():
    s = SalienceEngine()
    s.observe("cold")
    assert s.top(1)[0][0] == "cold"


def test_observe_increases_weight():
    s = SalienceEngine()
    s.observe("fire")
    assert s.weights["fire"] > 0.0


def test_observe_novelty_decreases_with_repetition():
    s = SalienceEngine()
    s.observe("fire")
    w1 = s.weights["fire"]
    s.observe("fire")
    w2 = s.weights["fire"]
    # Second observation adds less than the first (novelty decay)
    assert w2 - w1 < w1


def test_top_k_returns_at_most_k():
    s = SalienceEngine()
    for label in ("a", "b", "c", "d"):
        s.observe(label)
    assert len(s.top(2)) == 2


def test_top_sorted_descending():
    s = SalienceEngine()
    s.observe("fire")
    s.observe("fire")
    s.observe("food")
    top = s.top(2)
    assert top[0][1] >= top[1][1]


def test_top_empty_when_no_observations():
    s = SalienceEngine()
    assert s.top(3) == []


def test_decay_reduces_all_weights():
    s = SalienceEngine()
    s.observe("fire")
    s.observe("food")
    before = dict(s.weights)
    s.decay(rate=0.1)
    for label in before:
        assert s.weights.get(label, 0.0) < before[label]


def test_decay_removes_near_zero_weights():
    s = SalienceEngine()
    s.weights["ghost"] = 0.005
    s.decay(rate=0.5)
    assert "ghost" not in s.weights


def test_serialization_roundtrip():
    s = SalienceEngine()
    s.observe("fire")
    s.observe("cold")
    d = s.to_dict()
    s2 = SalienceEngine.from_dict(d)
    assert s2.weights == s.weights
    assert list(s2.trace) == list(s.trace)
