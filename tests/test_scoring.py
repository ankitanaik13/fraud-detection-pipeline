from src.common.scoring import risk_level, score_triggered_rules


def test_precision_weighted_risk_score():
    score, counts = score_triggered_rules(["velocity,amount", "geo", "velocity"])
    assert score == 3.55
    assert counts == {"amount": 1, "geo": 1, "velocity": 2}
    assert risk_level(score) == "high"


def test_empty_history_has_no_risk():
    score, counts = score_triggered_rules([])
    assert score == 0
    assert counts == {"amount": 0, "geo": 0, "velocity": 0}
    assert risk_level(score) == "none"


def test_unknown_rule_is_counted_but_not_scored():
    score, counts = score_triggered_rules(["unknown"])
    assert score == 0
    assert counts["unknown"] == 1
