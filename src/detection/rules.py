"""Pure, unit-testable fraud-rule calculations used by the Spark adapter."""

import math
from dataclasses import dataclass

from src.common.geo import haversine_km


@dataclass(frozen=True)
class RuleConfig:
    velocity_min_observations: int = 5
    velocity_min_ratio: float = 1.2
    velocity_z_threshold: float = 2.0
    amount_min_observations: int = 3
    amount_spike_multiplier: float = 5.0
    impossible_speed_kmh: float = 900.0


DEFAULT_CONFIG = RuleConfig()


def velocity_signal(
    current_count: int,
    baseline_observations: int,
    baseline_mean: float,
    baseline_m2: float = 0.0,
    config: RuleConfig = DEFAULT_CONFIG,
) -> tuple[bool, float, float]:
    """Score a live count against completed windows using a stable z-score.

    ``baseline_m2`` is Welford's running sum of squared deviations. The standard
    deviation is floored at the Poisson expectation ``sqrt(mean)`` so a handful
    of identical low-count windows cannot produce an infinite or unstable score.
    """
    ratio = current_count / baseline_mean if baseline_mean > 0 else 0.0
    sample_variance = (
        baseline_m2 / (baseline_observations - 1) if baseline_observations > 1 else 0.0
    )
    effective_std = max(math.sqrt(max(sample_variance, 0.0)), math.sqrt(baseline_mean), 1.0)
    z_score = (current_count - baseline_mean) / effective_std if baseline_mean > 0 else 0.0
    flagged = (
        baseline_mean > 0
        and baseline_observations >= config.velocity_min_observations
        and ratio > config.velocity_min_ratio
        and z_score > config.velocity_z_threshold
    )
    return flagged, ratio, z_score


def amount_signal(
    amount: float,
    historical_count: int,
    historical_mean: float,
    config: RuleConfig = DEFAULT_CONFIG,
) -> tuple[bool, float]:
    ratio = amount / historical_mean if historical_mean > 0 else 0.0
    flagged = (
        historical_mean > 0
        and
        historical_count >= config.amount_min_observations
        and amount > historical_mean * config.amount_spike_multiplier
    )
    return flagged, ratio


def geo_signal(
    previous_country: str | None,
    current_country: str,
    elapsed_ms: int | None,
    config: RuleConfig = DEFAULT_CONFIG,
) -> tuple[bool, float]:
    if previous_country is None or elapsed_ms is None or previous_country == current_country:
        return False, 0.0
    hours_elapsed = max(elapsed_ms / 3_600_000.0, 1e-6)
    speed = haversine_km(previous_country, current_country) / hours_elapsed
    return speed > config.impossible_speed_kmh, speed
