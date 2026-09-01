"""Shared rule calibration and account-risk scoring.

Weights are the Wilson 95% lower confidence bounds for precision from the
quality-gated v2 release run, rather than optimistic point estimates.
"""

RULE_PRECISION = {
    "amount": 0.9417,
    "geo": 0.8957,
    "velocity": 0.8586,
}

SCORING_CALIBRATION_VERSION = "v2-release-20260901-precision-ci95-lower"

RISK_LEVEL_THRESHOLDS = (
    (3.0, "high"),
    (1.0, "medium"),
    (0.0, "low"),
)


def risk_level(score: float) -> str:
    if score <= 0:
        return "none"
    for threshold, label in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


def score_triggered_rules(rows: list[str]) -> tuple[float, dict[str, int]]:
    counts = {rule: 0 for rule in RULE_PRECISION}
    score = 0.0
    for triggered_rules in rows:
        for rule in filter(None, triggered_rules.split(",")):
            counts[rule] = counts.get(rule, 0) + 1
            score += RULE_PRECISION.get(rule, 0.0)
    return round(score, 2), counts
