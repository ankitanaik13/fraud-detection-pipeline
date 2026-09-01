"""Dependency-free precision/recall metrics for detector output."""

import math
from dataclasses import asdict, dataclass

RULE_LABELS = {
    "amount": "amount_spike",
    "geo": "impossible_travel",
    "velocity": "velocity_burst",
}


@dataclass(frozen=True)
class RuleMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    precision_ci95: tuple[float, float]
    recall_ci95: tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def metrics_from_predictions(expected: list[bool], predicted: list[bool]) -> RuleMetrics:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    tp = sum(truth and guess for truth, guess in zip(expected, predicted, strict=True))
    fp = sum(not truth and guess for truth, guess in zip(expected, predicted, strict=True))
    fn = sum(truth and not guess for truth, guess in zip(expected, predicted, strict=True))
    tn = sum(not truth and not guess for truth, guess in zip(expected, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RuleMetrics(
        tp,
        fp,
        fn,
        tn,
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        _wilson_interval(tp, tp + fp),
        _wilson_interval(tp, tp + fn),
    )


def expected_for_rule(row: dict, rule: str, geo_threshold_kmh: float = 900.0) -> bool:
    label_match = row.get("injected_label") == RULE_LABELS[rule]
    if rule == "geo" and label_match:
        # Straddle injections deliberately include below-threshold examples.
        return float(row.get("implied_speed_kmh") or 0.0) > geo_threshold_kmh
    return label_match


def evaluate_rows(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for rule in RULE_LABELS:
        expected = [expected_for_rule(row, rule) for row in rows]
        predicted = [bool(row.get(f"flag_{rule}")) for row in rows]
        output[rule] = metrics_from_predictions(expected, predicted).to_dict()
    expected_any = [any(expected_for_rule(row, rule) for rule in RULE_LABELS) for row in rows]
    predicted_any = [bool(row.get("is_flagged")) for row in rows]
    output["overall"] = metrics_from_predictions(expected_any, predicted_any).to_dict()
    return output


def evaluate_quality_gate(
    metrics: dict[str, dict],
    *,
    min_positive_examples: int = 20,
    min_precision: float = 0.5,
    min_recall: float = 0.5,
) -> dict:
    """Apply explicit release criteria without hiding insufficient sample sizes."""
    rules = {}
    for rule in RULE_LABELS:
        values = metrics[rule]
        positive_examples = values["true_positive"] + values["false_negative"]
        reasons = []
        if positive_examples < min_positive_examples:
            reasons.append(
                f"only {positive_examples} positive examples; need {min_positive_examples}"
            )
        if values["precision"] < min_precision:
            reasons.append(f"precision {values['precision']:.3f} < {min_precision:.3f}")
        if values["recall"] < min_recall:
            reasons.append(f"recall {values['recall']:.3f} < {min_recall:.3f}")
        rules[rule] = {
            "passed": not reasons,
            "positive_examples": positive_examples,
            "reasons": reasons,
        }
    return {
        "passed": all(result["passed"] for result in rules.values()),
        "requirements": {
            "min_positive_examples_per_rule": min_positive_examples,
            "min_precision": min_precision,
            "min_recall": min_recall,
        },
        "rules": rules,
    }


def velocity_threshold_sweep(
    rows: list[dict], thresholds: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
) -> list[dict]:
    """Rescore persisted continuous velocity signals without rerunning Spark."""
    expected = [row.get("injected_label") == RULE_LABELS["velocity"] for row in rows]
    output = []
    for threshold in thresholds:
        predicted = [
            float(row.get("velocity_z_score") or 0.0) > threshold
            and float(row.get("velocity_ratio_to_baseline") or 0.0) > 1.2
            for row in rows
        ]
        output.append({"z_threshold": threshold, **metrics_from_predictions(expected, predicted).to_dict()})
    return output
