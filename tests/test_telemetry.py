import json
from metabolic_intelligence_lab.core.telemetry import Telemetry


def _populated() -> Telemetry:
    t = Telemetry()
    t.log(0, "scout", "tick", {"energy_used": 0.05, "top_label": "fire", "threshold": 1.2, "frontier": []})
    t.log(0, "scout", "plan:fire:collect_wood", {"tool": "collect_wood", "cost": 0.02})
    t.log(0, "scout", "plan_reward:fire", {
        "total_reward": 0.7, "ces_like": 2.5, "survival": 0.8, "time_cost": 0.03
    })
    t.log(1, "scout", "tick", {"energy_used": 0.30, "top_label": "fire", "threshold": 1.3, "frontier": []})
    t.log(1, "scout", "plan_reward:shelter", {
        "total_reward": 0.5, "ces_like": 1.8, "survival": 0.6, "time_cost": 0.06
    })
    return t


def test_telemetry_log_appends():
    t = Telemetry()
    t.log(0, "agent", "tick")
    assert len(t.rows) == 1
    assert t.rows[0]["tag"] == "tick"


def test_filter_by_tag_prefix():
    t = _populated()
    plan_rows = t.filter_by_tag("plan_reward:")
    assert len(plan_rows) == 2
    assert all(r["tag"].startswith("plan_reward:") for r in plan_rows)


def test_filter_by_tag_exact():
    t = _populated()
    tick_rows = t.filter_by_tag("tick", prefix=False)
    assert len(tick_rows) == 2
    assert all(r["tag"] == "tick" for r in tick_rows)


def test_filter_by_tag_no_match():
    t = _populated()
    assert t.filter_by_tag("nonexistent:") == []


def test_summary_stats_counts():
    t = _populated()
    s = t.summary_stats()
    assert s["n_ticks"] == 2
    assert s["n_plans"] == 2


def test_summary_stats_avg_reward():
    t = _populated()
    s = t.summary_stats()
    assert abs(s["avg_reward"] - 0.6) < 1e-6


def test_summary_stats_survival_rate():
    t = _populated()
    s = t.summary_stats()
    # survival values: 0.8 and 0.6; both >= 0.5 → rate = 1.0
    assert s["survival_rate"] == 1.0


def test_summary_stats_label_distribution():
    t = _populated()
    s = t.summary_stats()
    assert s["label_distribution"].get("fire") == 2


def test_summary_stats_goal_distribution():
    t = _populated()
    s = t.summary_stats()
    assert "fire" in s["goal_distribution"]
    assert "shelter" in s["goal_distribution"]


def test_write_csv(tmp_path):
    t = _populated()
    csv_path = tmp_path / "out.csv"
    t.write_csv(csv_path)
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) > 1   # header + data rows


def test_write_json(tmp_path):
    t = _populated()
    json_path = tmp_path / "out.json"
    t.write_json(json_path)
    rows = json.loads(json_path.read_text())
    assert len(rows) == len(t.rows)


def test_write_replay(tmp_path):
    t = _populated()
    replay_path = tmp_path / "replay.json"
    t.write_replay(replay_path, config_name="baseline", seed=42)
    manifest = json.loads(replay_path.read_text())
    assert manifest["config_name"] == "baseline"
    assert manifest["seed"] == 42
    assert "stats" in manifest


def test_write_csv_empty_is_noop(tmp_path):
    t = Telemetry()
    csv_path = tmp_path / "empty.csv"
    t.write_csv(csv_path)
    assert not csv_path.exists()
