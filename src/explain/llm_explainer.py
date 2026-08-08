"""Calls NVIDIA NIM (meta/llama-3.1-70b-instruct) to turn each flagged transaction in
Postgres into a one-sentence plain-English explanation, grounded strictly in the rule(s)
src/streaming/fraud_detector.py actually tripped and that transaction's specific values —
never raw guesswork. Confidence language is calibrated to each rule's measured precision
(see README): stated directly for amount (~100% precision), hedged for geo (~50%), hedged
strongly for velocity (~26%, more often a false positive than not).

Writes results back to flagged_transactions.explanation. Safe to re-run — only rows with
explanation IS NULL are processed.
"""

import argparse
import os
import random
import sys
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_POSTGRES_URL = "postgresql://fraud:fraud@localhost:5432/frauddb"

# Free-tier NIM rate limit is ~40 req/min; pace proactively under that rather than relying
# purely on reactive 429 retries.
MIN_SECONDS_BETWEEN_CALLS = 60.0 / 35.0
MAX_RETRIES = 6
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIStatusError)

# Measured against the 20-min/10k-event soak run documented in README.md. Passed to the
# model as context so it calibrates confidence language instead of asserting every flag
# with the same certainty.
RULE_PRECISION = {
    "amount": 1.00,
    "geo": 0.50,
    "velocity": 0.26,
}

# Country codes used by src/producer/generate_transactions.py. Several overlap with common
# US state abbreviations (DE, GA, ...) — an early test had the model read "DE" as Delaware
# instead of Germany, so we spell out the full name ourselves rather than let it guess.
COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "DE": "Germany", "FR": "France",
    "JP": "Japan", "AU": "Australia", "BR": "Brazil", "IN": "India", "CA": "Canada",
    "SG": "Singapore", "NL": "Netherlands", "MX": "Mexico", "ZA": "South Africa",
    "AE": "United Arab Emirates", "KR": "South Korea",
}


def country_label(code: str) -> str:
    name = COUNTRY_NAMES.get(code)
    return f"{code} ({name})" if name else code

SYSTEM_PROMPT = """You are a fraud-analyst assistant. You write exactly ONE plain-English \
sentence explaining why an automated rule engine flagged a transaction.

Rules:
- Ground the explanation ONLY in the rule name(s) and numeric facts given to you. Never \
invent a reason, signal, or detail that isn't in the provided facts. Use exactly the \
country names given to you — never reinterpret a code or abbreviation as something else.
- Calibrate your confidence language to the stated precision of each rule that fired:
  - precision >= 0.9: state the finding directly and plainly.
  - precision 0.4-0.7: use hedged language ("may indicate", "is consistent with").
  - precision < 0.4: hedge strongly and say explicitly that this rule is frequently a \
false positive in testing, e.g. "...though this signal is unreliable and more often than \
not a false positive in testing."
- If multiple rules fired, cover each one, calibrated independently.
- Output ONLY the one sentence. No preamble, no quotes, no markdown."""


def rule_detail(rule: str, row: dict) -> str:
    if rule == "amount":
        return (
            f"amount ${row['amount']:.2f} is {row['amount_ratio_to_avg']:.1f}x this "
            f"account's historical average transaction amount"
        )
    if rule == "geo":
        return (
            f"implied travel speed from this account's previous transaction location to "
            f"{country_label(row['location'])} was {row['implied_speed_kmh']:,.0f} km/h"
        )
    if rule == "velocity":
        return (
            f"{row['velocity_count_5min']} transactions in the trailing 5 minutes for this "
            f"account, {row['velocity_ratio_to_baseline']:.1f}x its own average count per "
            f"completed 5-minute window"
        )
    raise ValueError(f"unknown rule: {rule}")


def build_user_message(row: dict) -> str:
    rules = row["triggered_rules"].split(",")
    facts = "\n".join(
        f"- {rule} (precision in testing: {RULE_PRECISION[rule]:.0%}): {rule_detail(rule, row)}"
        for rule in rules
    )
    return (
        f"Transaction: account {row['account_id']}, ${row['amount']:.2f} at "
        f"{row['merchant']} in {country_label(row['location'])} on {row['event_timestamp']}.\n\n"
        f"Rule(s) triggered:\n{facts}\n\n"
        f"Write the one-sentence explanation. Use the country names given, not the raw "
        f"two-letter codes, and do not reinterpret them as anything else (e.g. US states)."
    )


def call_with_retry(client: OpenAI, row: dict) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=NIM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(row)},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except RETRYABLE_EXCEPTIONS as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
            delay = float(retry_after) if retry_after else min(2 ** attempt * 2, 60)
            delay += random.uniform(0, 1)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {type(exc).__name__}, backing off {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE flagged_transactions ADD COLUMN IF NOT EXISTS explanation TEXT")
        cur.execute("ALTER TABLE flagged_transactions ADD COLUMN IF NOT EXISTS explained_at TIMESTAMPTZ")
    conn.commit()


def fetch_unexplained(conn, limit: int, transaction_ids: list[str] = None) -> list[dict]:
    columns = """transaction_id, account_id, amount, merchant, location, event_timestamp,
                 triggered_rules, velocity_count_5min, velocity_ratio_to_baseline,
                 amount_ratio_to_avg, implied_speed_kmh"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if transaction_ids:
            cur.execute(
                f"SELECT {columns} FROM flagged_transactions WHERE transaction_id = ANY(%s)",
                (transaction_ids,),
            )
        else:
            cur.execute(
                f"""
                SELECT {columns}
                FROM flagged_transactions
                WHERE explanation IS NULL
                ORDER BY event_timestamp
                LIMIT %s
                """,
                (limit,),
            )
        return cur.fetchall()


def write_explanation(conn, transaction_id: str, explanation: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE flagged_transactions SET explanation = %s, explained_at = now() WHERE transaction_id = %s",
            (explanation, transaction_id),
        )
    conn.commit()


def run(limit: int, postgres_url: str, transaction_ids: list[str] = None) -> None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY not set (check your .env)")

    client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key, timeout=30.0, max_retries=0)
    conn = psycopg2.connect(postgres_url)
    ensure_schema(conn)

    rows = fetch_unexplained(conn, limit, transaction_ids)
    print(f"Explaining {len(rows)} flagged transaction(s)...")

    last_call = 0.0
    for i, row in enumerate(rows, 1):
        elapsed = time.monotonic() - last_call
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

        explanation = call_with_retry(client, row)
        last_call = time.monotonic()
        write_explanation(conn, row["transaction_id"], explanation)

        print(f"[{i}/{len(rows)}] {row['triggered_rules']:<20} {row['transaction_id'][:8]}  {explanation}")

    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM explanations for flagged transactions.")
    parser.add_argument("--limit", type=int, default=50, help="max unexplained rows to process (default: 50)")
    parser.add_argument("--postgres-url", type=str, default=os.getenv("POSTGRES_URL") or DEFAULT_POSTGRES_URL)
    parser.add_argument(
        "--transaction-ids", type=str, default=None,
        help="comma-separated transaction_id(s) to (re-)explain, ignoring --limit and the explanation IS NULL filter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ids = [t.strip() for t in args.transaction_ids.split(",")] if args.transaction_ids else None
    run(limit=args.limit, postgres_url=args.postgres_url, transaction_ids=ids)
