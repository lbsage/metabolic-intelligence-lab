from metabolic_intelligence_lab.core.tasks import TaskQueue


def test_add_returns_string_id():
    q = TaskQueue()
    tid = q.add(trigger_time=5, label_if=None, min_salience=1.0, action="noop")
    assert isinstance(tid, str)
    assert len(tid) > 0


def test_due_by_time():
    q = TaskQueue()
    q.add(trigger_time=3, label_if=None, min_salience=0.0, action="do_it")
    ready = q.due(t=3, salience_top=[("fire", 2.0)])
    assert len(ready) == 1
    assert ready[0].action == "do_it"


def test_not_due_before_trigger_time():
    q = TaskQueue()
    q.add(trigger_time=5, label_if=None, min_salience=0.0, action="do_it")
    ready = q.due(t=4, salience_top=[("fire", 2.0)])
    assert ready == []


def test_due_consumed_from_queue():
    q = TaskQueue()
    q.add(trigger_time=1, label_if=None, min_salience=0.0, action="noop")
    q.due(t=1, salience_top=[])
    assert q.due(t=2, salience_top=[]) == []


def test_label_gate_blocks_wrong_label():
    q = TaskQueue()
    q.add(trigger_time=1, label_if="cold", min_salience=0.0, action="warm_up")
    ready = q.due(t=1, salience_top=[("fire", 2.0)])
    assert ready == []


def test_label_gate_passes_correct_label():
    q = TaskQueue()
    q.add(trigger_time=1, label_if="cold", min_salience=0.0, action="warm_up")
    ready = q.due(t=1, salience_top=[("cold", 2.0)])
    assert len(ready) == 1


def test_salience_gate_blocks_low_salience():
    q = TaskQueue()
    q.add(trigger_time=1, label_if=None, min_salience=2.0, action="expensive")
    ready = q.due(t=1, salience_top=[("fire", 1.0)])
    assert ready == []


def test_salience_gate_passes_high_enough():
    q = TaskQueue()
    q.add(trigger_time=1, label_if=None, min_salience=1.5, action="expensive")
    ready = q.due(t=1, salience_top=[("fire", 2.0)])
    assert len(ready) == 1


def test_multiple_tasks_due_at_same_time():
    q = TaskQueue()
    q.add(trigger_time=2, label_if=None, min_salience=0.0, action="a")
    q.add(trigger_time=2, label_if=None, min_salience=0.0, action="b")
    q.add(trigger_time=5, label_if=None, min_salience=0.0, action="c")
    ready = q.due(t=2, salience_top=[("x", 1.0)])
    assert len(ready) == 2
    assert len(q.queue) == 1   # "c" remains
