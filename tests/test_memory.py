import numpy as np
from metabolic_intelligence_lab.core.memory import GeometricMemory


def test_memory_retrieve():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "x")
    assert gm.retrieve(np.array([1.0, 0.0]), threshold=0.5)[0][1] == "x"


def test_store_and_retrieve_by_label():
    gm = GeometricMemory(dim=3)
    gm.store(np.array([1.0, 0.0, 0.0]), "fire")
    gm.store(np.array([0.0, 1.0, 0.0]), "food")
    hits = gm.retrieve(np.array([1.0, 0.0, 0.0]), threshold=0.5)
    labels = [h[1] for h in hits]
    assert "fire" in labels
    assert "food" not in labels


def test_retrieve_returns_empty_below_threshold():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "x", strength=0.01)
    assert gm.retrieve(np.array([0.0, 1.0]), threshold=0.9) == []


def test_retrieve_sorted_by_weighted_similarity():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "strong", strength=2.0)
    gm.store(np.array([1.0, 0.0]), "weak", strength=0.5)
    hits = gm.retrieve(np.array([1.0, 0.0]), threshold=0.4)
    assert hits[0][1] == "strong"


def test_decay_reduces_strength():
    gm = GeometricMemory(dim=2, memory_decay=0.1)
    gm.store(np.array([1.0, 0.0]), "x", strength=1.0)
    gm.decay()
    assert gm.items[0]["strength"] < 1.0


def test_decay_uses_memory_decay_rate():
    gm = GeometricMemory(dim=2, memory_decay=0.2)
    gm.store(np.array([1.0, 0.0]), "x", strength=1.0)
    gm.decay()
    assert abs(gm.items[0]["strength"] - 0.8) < 1e-9


def test_decay_custom_rate_overrides():
    gm = GeometricMemory(dim=2, memory_decay=0.1)
    gm.store(np.array([1.0, 0.0]), "x", strength=1.0)
    gm.decay(rate=0.5)
    assert abs(gm.items[0]["strength"] - 0.5) < 1e-9


def test_reinforce_increases_hit_strength():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "fire", strength=1.0)
    before = gm.items[0]["strength"]
    gm.reinforce(["fire"], alpha=0.1, beta=0.0)
    assert gm.items[0]["strength"] > before


def test_reinforce_decays_miss_strength():
    gm = GeometricMemory(dim=2)
    gm.store(np.array([1.0, 0.0]), "fire", strength=1.0)
    gm.store(np.array([0.0, 1.0]), "food", strength=1.0)
    gm.reinforce(["fire"], alpha=0.0, beta=0.1)
    food_item = next(m for m in gm.items if m["label"] == "food")
    assert food_item["strength"] < 1.0


def test_serialization_roundtrip():
    gm = GeometricMemory(dim=3, memory_decay=0.05)
    gm.store(np.array([1.0, 0.0, 0.0]), "fire", strength=0.8)
    gm.store(np.array([0.0, 1.0, 0.0]), "food", strength=0.6)
    d = gm.to_dict()
    gm2 = GeometricMemory.from_dict(d)
    assert gm2.dim == 3
    assert gm2.memory_decay == 0.05
    assert len(gm2.items) == 2
    labels = [m["label"] for m in gm2.items]
    assert "fire" in labels and "food" in labels
