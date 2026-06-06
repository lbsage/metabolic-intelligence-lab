from metabolic_intelligence_lab.core.bus import DUSEBus, EnergyBudget, Message


# -- EnergyBudget ----------------------------------------------------------

def test_energy_budget_spend_within_limit():
    b = EnergyBudget(total=1.0, reserve=0.3)
    assert b.can_spend(0.5)
    assert b.spend(0.5)
    assert abs(b.used - 0.5) < 1e-9


def test_energy_budget_blocked_by_reserve():
    b = EnergyBudget(total=1.0, reserve=0.3)
    assert not b.can_spend(0.8)   # 0.8 > total - reserve = 0.7
    assert not b.spend(0.8)
    assert b.used == 0.0


def test_energy_budget_reset():
    b = EnergyBudget(total=1.0, reserve=0.3)
    b.spend(0.4)
    b.reset()
    assert b.used == 0.0


# -- DUSEBus point-to-point -----------------------------------------------

def _msg(src="a", dst="b", topic="ping", priority=0.5):
    return Message(src=src, dst=dst, topic=topic, payload={}, priority=priority)


def test_bus_point_to_point():
    bus = DUSEBus()
    bus.register("a")
    bus.register("b")
    bus.send(_msg(src="a", dst="b"))
    msgs = bus.recv("b")
    assert len(msgs) == 1
    assert msgs[0].topic == "ping"


def test_bus_sender_does_not_receive_own_message():
    bus = DUSEBus()
    bus.register("a")
    bus.register("b")
    bus.send(_msg(src="a", dst="b"))
    assert bus.recv("a") == []


def test_bus_broadcast():
    bus = DUSEBus()
    for aid in ("a", "b", "c"):
        bus.register(aid)
    bus.send(_msg(src="a", dst="*", topic="broadcast"))
    for aid in ("a", "b", "c"):
        msgs = bus.recv(aid)
        assert any(m.topic == "broadcast" for m in msgs), f"{aid} should receive broadcast"


def test_bus_unknown_dst_silently_dropped():
    bus = DUSEBus()
    bus.register("a")
    bus.send(_msg(dst="nobody"))   # should not raise
    assert bus.recv("a") == []


# -- Priority ordering -----------------------------------------------------

def test_bus_recv_priority_order():
    bus = DUSEBus()
    bus.register("b")
    bus.send(_msg(dst="b", topic="low", priority=0.1))
    bus.send(_msg(dst="b", topic="high", priority=0.9))
    bus.send(_msg(dst="b", topic="mid", priority=0.5))
    msgs = bus.recv("b")
    assert msgs[0].topic == "high"
    assert msgs[-1].topic == "low"


# -- Subscribe / filter ----------------------------------------------------

def test_bus_subscribe_filters_topic():
    bus = DUSEBus()
    bus.register("a")
    bus.subscribe("a", "reason:")
    bus.send(_msg(dst="a", topic="reason:fire"))
    bus.send(_msg(dst="a", topic="heartbeat"))
    msgs = bus.recv("a")
    assert len(msgs) == 1
    assert msgs[0].topic == "reason:fire"


def test_bus_unsubscribe_restores_all_delivery():
    bus = DUSEBus()
    bus.register("a")
    bus.subscribe("a", "reason:")
    bus.unsubscribe("a")
    bus.send(_msg(dst="a", topic="heartbeat"))
    assert len(bus.recv("a")) == 1


# -- History ---------------------------------------------------------------

def test_bus_history_records_all_sent():
    bus = DUSEBus()
    bus.register("a")
    bus.register("b")
    bus.send(_msg(src="a", dst="b", topic="t1"))
    bus.send(_msg(src="a", dst="b", topic="t2"))
    h = bus.history()
    assert len(h) == 2


def test_bus_history_topic_filter():
    bus = DUSEBus()
    bus.register("a")
    bus.register("b")
    bus.send(_msg(dst="b", topic="plan:fire"))
    bus.send(_msg(dst="b", topic="heartbeat"))
    plan_msgs = bus.history(topic_prefix="plan:")
    assert len(plan_msgs) == 1
    assert plan_msgs[0].topic == "plan:fire"


# -- Stats -----------------------------------------------------------------

def test_bus_stats_counts():
    bus = DUSEBus()
    bus.register("a")
    bus.register("b")
    bus.send(_msg(src="a", dst="b", topic="ping"))
    bus.send(_msg(src="a", dst="*", topic="broadcast"))
    stats = bus.stats()
    assert stats["sent_total"] == 2
    assert stats["topic_counts"]["ping"] == 1
    assert stats["topic_counts"]["broadcast"] == 1
    assert "a" in stats["registered_agents"]
