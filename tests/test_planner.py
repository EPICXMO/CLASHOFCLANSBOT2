from planner.rules import RulePlanner


def test_enemy_push_detected_true():
    rp = RulePlanner()
    ui = {
        "yolo": {
            "troop": [(100, 100, 120, 140)],
            "my_tower": [(110, 110, 140, 160)],
        }
    }
    assert rp.enemy_push_detected(ui) is True


def test_bandit_arms_expand_and_update():
    rp = RulePlanner()
    rp.bandit.ensure_arms(6)
    assert rp.bandit.k == 6
    rp.record_outcome(5, 10.0)
    assert rp.bandit.values[5] > 0

