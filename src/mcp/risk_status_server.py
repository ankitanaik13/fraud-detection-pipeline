"""MCP server exposing one tool, get_account_risk_status, that queries Postgres for an
account's flagged-transaction history and returns a risk score, a per-rule breakdown, and
the recent flagged transactions (with LLM explanation where available).

The risk score weights each triggered rule by its measured precision (see README) rather
than counting flags equally — an amount flag (~100% precision) should move the needle far
more than a velocity flag (~26% precision, more often noise than signal).

Run: python src/mcp/risk_status_server.py  (stdio transport)
"""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv()

DEFAULT_POSTGRES_URL = "postgresql://fraud:fraud@localhost:5432/frauddb"
RECENT_FLAGS_LIMIT = 20

# Same precision figures used by src/explain/llm_explainer.py, measured against the
# 20-min/10k-event soak run documented in README.md.
RULE_PRECISION = {
    "amount": 1.00,
    "geo": 0.50,
    "velocity": 0.26,
}

RISK_LEVEL_THRESHOLDS = (
    (3.0, "high"),
    (1.0, "medium"),
    (0.0, "low"),
)

server = MCPServer("risk-status")


def get_postgres_url() -> str:
    return os.getenv("POSTGRES_URL") or DEFAULT_POSTGRES_URL


def risk_level(score: float) -> str:
    if score <= 0:
        return "none"
    for threshold, label in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


@server.tool()
def get_account_risk_status(account_id: str) -> dict:
    """Return current risk status for an account: risk score, per-rule flag breakdown, and
    recent flagged transactions with their triggered rule(s) and LLM explanation."""
    conn = psycopg2.connect(get_postgres_url())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT transaction_id, account_id, amount, merchant, location, event_timestamp,
                       injected_label, triggered_rules, explanation
                FROM flagged_transactions
                WHERE account_id = %s
                ORDER BY event_timestamp DESC
                LIMIT %s
                """,
                (account_id, RECENT_FLAGS_LIMIT),
            )
            recent = cur.fetchall()

            cur.execute(
                "SELECT triggered_rules FROM flagged_transactions WHERE account_id = %s",
                (account_id,),
            )
            all_triggered = cur.fetchall()

            cur.execute(
                "SELECT count(*) AS n FROM processed_transactions WHERE account_id = %s",
                (account_id,),
            )
            total_processed = cur.fetchone()["n"]
    finally:
        conn.close()

    flags_by_rule = {rule: 0 for rule in RULE_PRECISION}
    risk_score = 0.0
    for row in all_triggered:
        for rule in row["triggered_rules"].split(","):
            flags_by_rule[rule] = flags_by_rule.get(rule, 0) + 1
            risk_score += RULE_PRECISION.get(rule, 0.0)

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
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level(risk_score),
        "total_flags": len(all_triggered),
        "total_processed_transactions": total_processed,
        "flags_by_rule": flags_by_rule,
        "last_explanation": recent_flags[0]["explanation"] if recent_flags else None,
        "recent_flags": recent_flags,
    }


if __name__ == "__main__":
    server.run()
