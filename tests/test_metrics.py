import pytest

from src.evaluation.metrics import (
    evaluate_quality_gate,
    evaluate_rows,
    expected_for_rule,
    metrics_from_predictions,
    velocity_threshold_sweep,
)


def test_metrics_are_reproducible():
    metrics = metrics_from_predictions([True, True, False, False], [True, False, True, False])
    assert metrics.to_dict() == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "precision_ci95": (0.0945, 0.9055),
        "recall_ci95": (0.0945, 0.9055),
    }


def test_geo_straddle_truth_uses_achieved_speed():
    below = {"injected_label": "impossible_travel", "implied_speed_kmh": 899.9}
    above = {"injected_label": "impossible_travel", "implied_speed_kmh": 900.1}
    assert expected_for_rule(below, "geo") is False
    assert expected_for_rule(above, "geo") is True


def test_evaluate_rows_returns_each_rule_and_overall():
    rows = [
        {
            "injected_label": "amount_spike",
            "flag_amount": True,
            "flag_geo": False,
            "flag_velocity": False,
            "is_flagged": True,
            "implied_speed_kmh": 0,
        }
    ]
    result = evaluate_rows(rows)
    assert set(result) == {"amount", "geo", "velocity", "overall"}
    assert result["amount"]["precision"] == 1.0


def test_overall_truth_excludes_below_threshold_geo_straddle():
    rows = [
        {
            "injected_label": "impossible_travel",
            "flag_amount": False,
            "flag_geo": False,
            "flag_velocity": False,
            "is_flagged": False,
            "implied_speed_kmh": 850,
        }
    ]
    assert evaluate_rows(rows)["overall"]["false_negative"] == 0


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        metrics_from_predictions([True], [])


def test_quality_gate_reports_low_support_and_metric_failures():
    rows = [
        {
            "injected_label": "amount_spike",
            "flag_amount": True,
            "flag_geo": False,
            "flag_velocity": False,
            "is_flagged": True,
            "implied_speed_kmh": 0,
        }
    ]
    gate = evaluate_quality_gate(evaluate_rows(rows), min_positive_examples=2)
    assert gate["passed"] is False
    assert "only 1 positive examples" in gate["rules"]["amount"]["reasons"][0]


def test_velocity_threshold_sweep_rescores_continuous_signal():
    rows = [
        {
            "injected_label": "velocity_burst",
            "velocity_z_score": 2.2,
            "velocity_ratio_to_baseline": 2.0,
        },
        {
            "injected_label": None,
            "velocity_z_score": 1.7,
            "velocity_ratio_to_baseline": 2.0,
        },
    ]
    sweep = velocity_threshold_sweep(rows, thresholds=(1.5, 2.0))
    assert sweep[0]["false_positive"] == 1
    assert sweep[1]["precision"] == 1.0
