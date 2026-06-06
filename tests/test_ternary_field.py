from metabolic_intelligence_lab.core.ternary_field import TernaryField


LABELS = ["food", "fire", "shelter"]


def test_initial_states_are_zero():
    tf = TernaryField(LABELS)
    assert all(s == 0 for s in tf.states.values())


def test_step_positive_input_drives_state_positive():
    tf = TernaryField(LABELS, coupling_gamma=0.0, coupling_decay=0.0)
    tf.coupling = {}   # zero out coupling so input drives state directly
    tf.step({"food": 1.0, "fire": 0.0, "shelter": 0.0}, hysteresis=0.5)
    assert tf.states["food"] == 1


def test_step_negative_input_drives_state_negative():
    tf = TernaryField(LABELS, coupling_gamma=0.0, coupling_decay=0.0)
    tf.coupling = {}
    tf.step({"food": -1.0, "fire": 0.0, "shelter": 0.0}, hysteresis=0.5)
    assert tf.states["food"] == -1


def test_step_weak_input_stays_zero():
    tf = TernaryField(LABELS, coupling_gamma=0.0, coupling_decay=0.0)
    tf.coupling = {}
    tf.step({"food": 0.1, "fire": 0.0, "shelter": 0.0}, hysteresis=0.5)
    assert tf.states["food"] == 0


def test_modulate_positive_state_increases_salience():
    tf = TernaryField(LABELS)
    tf.states = {"food": 1, "fire": 0, "shelter": 0}
    weights = {"food": 1.0, "fire": 0.5, "shelter": 0.0}
    out = tf.modulate(weights, gain=0.1)
    assert out["food"] > weights["food"]
    assert out["fire"] == weights["fire"]


def test_modulate_negative_state_decreases_salience():
    tf = TernaryField(LABELS)
    tf.states = {"food": -1, "fire": 0, "shelter": 0}
    weights = {"food": 0.5, "fire": 0.0, "shelter": 0.0}
    out = tf.modulate(weights, gain=0.1)
    assert out["food"] < weights["food"]


def test_modulate_clamps_to_zero():
    tf = TernaryField(LABELS)
    tf.states = {"food": -1, "fire": 0, "shelter": 0}
    weights = {"food": 0.05, "fire": 0.0, "shelter": 0.0}
    out = tf.modulate(weights, gain=0.1)
    assert out["food"] >= 0.0


def test_reinforce_couplings_strengthens_hit_pairs():
    tf = TernaryField(["a", "b"])
    tf.coupling[("a", "b")] = 0.1
    before = tf.coupling[("a", "b")]
    tf.reinforce_couplings(["a", "b"], gamma=0.1, decay=0.0)
    assert tf.coupling[("a", "b")] > before


def test_reinforce_couplings_decays_miss_pairs():
    tf = TernaryField(["a", "b", "c"])
    tf.coupling[("a", "b")] = 0.1
    tf.coupling[("a", "c")] = 0.1
    tf.coupling[("b", "c")] = 0.1
    tf.reinforce_couplings(["a"], gamma=0.0, decay=0.1)
    # No pair where both a and b (or c) hit
    for w in tf.coupling.values():
        assert w < 0.1 + 1e-9   # all should decay


def test_set_coupling_dynamics():
    tf = TernaryField(LABELS)
    tf.set_coupling_dynamics(gamma=0.05, decay=0.02)
    assert tf.coupling_gamma == 0.05
    assert tf.coupling_decay == 0.02


def test_serialization_roundtrip():
    tf = TernaryField(LABELS, coupling_gamma=0.03, coupling_decay=0.01)
    d = tf.to_dict()
    tf2 = TernaryField.from_dict(d)
    assert tf2.states == tf.states
    assert tf2.coupling_gamma == tf.coupling_gamma
    assert tf2.coupling == tf.coupling
