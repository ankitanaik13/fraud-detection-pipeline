"""MCP server exposing one tool, get_account_risk_status, that queries Postgres for an
account's flagged-transaction history and returns a risk score, a per-rule breakdown, and
the recent flagged transactions (with LLM explanation where available).

The risk score uses Wilson 95% lower precision bounds from the quality-gated v2 release
evaluation (see README), rather than optimistic point estimates or equal flag counts. The
response names the exact calibration version.

Run: python src/mcp/risk_status_server.py  (stdio transport)
"""

import os
import re

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from mcp.server import MCPServer

from src.common.scoring import (
    SCORING_CALIBRATION_VERSION,
    risk_level,
    score_triggered_rules,
)

load_dotenv()

DEFAULT_POSTGRES_URL = "postgresql://fraud:fraud@localhost:5432/frauddb"
RECENT_FLAGS_LIMIT = 20
DETECTOR_VERSION = os.getenv("DETECTOR_VERSION", "v2")

ACCOUNT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

server = MCPServer("risk-status")


def get_postgres_url() -> str:
    return os.getenv("POSTGRES_URL") or DEFAULT_POSTGRES_URL


@server.tool()
def get_account_risk_status(account_id: str) -> dict:
    """Return current risk status for an account: risk score, per-rule flag breakdown, and
    recent flagged transactions with their triggered rule(s) and LLM explanation."""
    if not ACCOUNT_ID_RE.fullmatch(account_id):
        raise ValueError("account_id must contain 1-64 letters, digits, underscores, or hyphens")

    conn = psycopg2.connect(get_postgres_url())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT transaction_id, account_id, amount, merchant, location, event_timestamp,
                       injected_label, triggered_rules, explanation
                FROM flagged_transactions
                WHERE account_id = %s AND detector_version = %s
                ORDER BY event_timestamp DESC
                LIMIT %s
                """,
                (account_id, DETECTOR_VERSION, RECENT_FLAGS_LIMIT),
            )
            recent = cur.fetchall()

            cur.execute(
                """SELECT triggered_rules FROM flagged_transactions
                   WHERE account_id = %s AND detector_version = %s""",
                (account_id, DETECTOR_VERSION),
            )
            all_triggered = cur.fetchall()

            cur.execute(
                """SELECT count(*) AS n FROM processed_transactions
                   WHERE account_id = %s AND detector_version = %s""",
                (account_id, DETECTOR_VERSION),
            )
            total_processed = cur.fetchone()["n"]
    finally:
        conn.close()

    risk_score, flags_by_rule = score_triggered_rules(
        [row["triggered_rules"] for row in all_triggered]
    )

    recent_flags = [
        {
            "transaction_id": row["transaction_id"],
            "event_timestamp": row["event_timestamp"].isoformat(),
            "amount": row["amount"],
            "merchant": row["merchant"],
            "location": row["location"],
            "triggered_rules": row["triggered_rules"].split(","),
            "injected_label": row["injected_label"],
            "explanation": row["explanation"],
        }
        for row in recent
    ]

    return {
        "account_id": account_id,
        "detector_version": DETECTOR_VERSION,
        "scoring_calibration": SCORING_CALIBRATION_VERSION,
        "risk_score": risk_score,
        "risk_level": risk_level(risk_score),
        "total_flags": len(all_triggered),
        "total_processed_transactions": total_processed,
        "flags_by_rule": flags_by_rule,
        "last_explanation": recent_flags[0]["explanation"] if recent_flags else None,
        "recent_flags": recent_flags,
    }


if __name__ == "__main__":
    server.run()
