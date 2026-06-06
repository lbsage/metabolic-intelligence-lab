from metabolic_intelligence_lab.core.world import WorldState


def test_has_returns_true_when_sufficient():
    w = WorldState()
    w.inventory["berries"] = 3
    assert w.has("berries", 3)


def test_has_returns_false_when_insufficient():
    w = WorldState()
    w.inventory["berries"] = 1
    assert not w.has("berries", 2)


def test_has_returns_false_for_missing_item():
    w = WorldState()
    assert not w.has("magic_wand", 1)


def test_add_increments_existing_item():
    w = WorldState()
    before = w.inventory.get("berries", 0)
    w.add("berries", 5)
    assert w.inventory["berries"] == before + 5


def test_add_creates_new_item():
    w = WorldState()
    w.add("diamond", 1)
    assert w.inventory["diamond"] == 1


def test_consume_deducts_and_returns_true():
    w = WorldState()
    w.inventory["sticks"] = 3
    result = w.consume("sticks", 2)
    assert result is True
    assert w.inventory["sticks"] == 1


def test_consume_fails_gracefully_when_insufficient():
    w = WorldState()
    w.inventory["sticks"] = 0
    result = w.consume("sticks", 1)
    assert result is False
    assert w.inventory["sticks"] == 0


def test_consume_missing_item_returns_false():
    w = WorldState()
    assert w.consume("unobtainium", 1) is False


def test_serialization_roundtrip():
    w = WorldState()
    w.weather = {"temp_c": -2, "status": "cold"}
    w.fire = {"lit": True, "duration_min": 30}
    w.hunger = 7
    w.inventory["sticks"] = 5
    d = w.to_dict()
    w2 = WorldState.from_dict(d)
    assert w2.weather["temp_c"] == -2
    assert w2.fire["lit"] is True
    assert w2.hunger == 7
    assert w2.inventory["sticks"] == 5


def test_from_dict_defaults_for_missing_keys():
    w = WorldState.from_dict({})
    assert isinstance(w.inventory, dict)
    assert isinstance(w.hunger, int)
