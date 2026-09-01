"""Evaluate persisted detector output and write a JSON report."""

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import psycopg2
import psycopg2.extras

from src.detection.rules import DEFAULT_CONFIG
from src.evaluation.metrics import evaluate_quality_gate, evaluate_rows, velocity_threshold_sweep


def fetch_rows(postgres_url: str, detector_version: str, run_id: str) -> list[dict]:
    with psycopg2.connect(postgres_url) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """SELECT injected_label, flag_amount, flag_geo, flag_velocity,
                      is_flagged, implied_speed_kmh, velocity_z_score,
                      velocity_ratio_to_baseline
               FROM processed_transactions
               WHERE detector_version = %s AND evaluation_run_id = %s""",
            (detector_version, run_id),
        )
        return [dict(row) for row in cur.fetchall()]


def main(
    postgres_url: str,
    output: Path,
    detector_version: str,
    run_id: str,
    *,
    fail_on_quality_gate: bool = False,
) -> None:
    rows = fetch_rows(postgres_url, detector_version, run_id)
    metrics = evaluate_rows(rows)
    gate = evaluate_quality_gate(metrics)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "detector_version": detector_version,
        "evaluation_run_id": run_id,
        "row_count": len(rows),
        "rule_config": asdict(DEFAULT_CONFIG),
        "metrics": metrics,
        "velocity_threshold_sweep": velocity_threshold_sweep(rows),
        "quality_gate": gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if fail_on_quality_gate and not gate["passed"]:
        raise SystemExit("Detector quality gate failed; see the report above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("POSTGRES_URL", "postgresql://fraud:fraud@localhost:5432/frauddb"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/latest_metrics.json"))
    parser.add_argument("--detector-version", default="v2")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fail-on-quality-gate", action="store_true")
    args = parser.parse_args()
    main(
        args.postgres_url,
        args.output,
        args.detector_version,
        args.run_id,
        fail_on_quality_gate=args.fail_on_quality_gate,
    )
