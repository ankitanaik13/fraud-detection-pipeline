
from src.common.geo import haversine_km
from src.detection.rules import RuleConfig, amount_signal, geo_signal, velocity_signal


def test_amount_rule_requires_history_and_uses_strict_boundary():
    assert amount_signal(1_000, 2, 100)[0] is False
    assert amount_signal(500, 3, 100)[0] is False
    assert amount_signal(501, 3, 100)[0] is True


def test_velocity_rule_uses_completed_window_baseline():
    assert velocity_signal(16, 0, 10)[0] is False
    assert velocity_signal(20, 4, 10, 3)[0] is False
    flagged, ratio, z_score = velocity_signal(21, 5, 10, 4)
    assert flagged is True
    assert ratio == 2.1
    assert z_score > 3


def test_velocity_variance_reduces_noisy_baseline_false_positive():
    stable = velocity_signal(20, 10, 10, 9)
    noisy = velocity_signal(20, 10, 10, 900)
    assert stable[0] is True
    assert noisy[0] is False


def test_geo_rule_straddles_threshold_exactly():
    distance = haversine_km("US", "GB")
    elapsed_for_900_kmh = int(distance / 900 * 3_600_000)
    config = RuleConfig(impossible_speed_kmh=900)

    below, _ = geo_signal("US", "GB", elapsed_for_900_kmh + 1_000, config)
    above, speed = geo_signal("US", "GB", elapsed_for_900_kmh - 1_000, config)

    assert below is False
    assert above is True
    assert speed > 900


def test_unknown_or_same_location_is_not_flagged():
    assert geo_signal(None, "US", 1_000) == (False, 0.0)
    assert geo_signal("US", "US", 1_000) == (False, 0.0)
    assert haversine_km("XX", "US") == 0.0


def test_zero_baselines_are_safe():
    assert amount_signal(100, 4, 0) == (False, 0.0)
    assert velocity_signal(10, 1, 0) == (False, 0.0, 0.0)
