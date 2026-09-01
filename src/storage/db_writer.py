"""Idempotent Postgres sink for Spark ``foreachBatch`` output.

Spark may retry a micro-batch after the database commit but before checkpoint
advancement. Plain JDBC append therefore provides at-least-once writes and can
duplicate transactions. This sink stages each batch and merges on the stable
``transaction_id`` primary key, making a replay safe.
"""

import os
from urllib.parse import unquote, urlparse

import psycopg2
from psycopg2 import sql

PROCESSED_TABLE = "processed_transactions"
FLAGGED_TABLE = "flagged_transactions"
DEFAULT_POSTGRES_URL = "postgresql://fraud:fraud@postgres:5432/frauddb"

COLUMNS = (
    "transaction_id",
    "detector_version",
    "evaluation_run_id",
    "account_id",
    "amount",
    "merchant",
    "location",
    "event_timestamp",
    "injected_label",
    "flag_velocity",
    "flag_amount",
    "flag_geo",
    "is_flagged",
    "triggered_rules",
    "velocity_count_5min",
    "velocity_ratio_to_baseline",
    "velocity_z_score",
    "amount_ratio_to_avg",
    "implied_speed_kmh",
)

_BASE_TABLE_DDL = """
    transaction_id TEXT PRIMARY KEY,
    detector_version TEXT NOT NULL,
    evaluation_run_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL CHECK (amount >= 0),
    merchant TEXT NOT NULL,
    location TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    injected_label TEXT,
    flag_velocity BOOLEAN NOT NULL,
    flag_amount BOOLEAN NOT NULL,
    flag_geo BOOLEAN NOT NULL,
    is_flagged BOOLEAN NOT NULL,
    triggered_rules TEXT NOT NULL,
    velocity_count_5min INTEGER NOT NULL,
    velocity_ratio_to_baseline DOUBLE PRECISION NOT NULL,
    velocity_z_score DOUBLE PRECISION NOT NULL,
    amount_ratio_to_avg DOUBLE PRECISION NOT NULL,
    implied_speed_kmh DOUBLE PRECISION NOT NULL
"""


def get_postgres_url() -> str:
    return os.getenv("POSTGRES_URL") or DEFAULT_POSTGRES_URL


def _parse_postgres_url(postgres_url: str):
    parsed = urlparse(postgres_url)
    if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname or not parsed.path:
        raise ValueError("POSTGRES_URL must be a valid postgres:// or postgresql:// URL")
    return parsed


def jdbc_config(postgres_url: str | None = None) -> tuple[str, dict[str, str]]:
    """Convert a Postgres URL to Spark JDBC configuration."""
    parsed = _parse_postgres_url(postgres_url or get_postgres_url())
    jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
    properties = {
        "user": unquote(parsed.username or "fraud"),
        "password": unquote(parsed.password or "fraud"),
        "driver": "org.postgresql.Driver",
    }
    return jdbc_url, properties


def ensure_schema(postgres_url: str | None = None) -> None:
    """Create stable tables, constraints, and query indexes before streaming."""
    with psycopg2.connect(postgres_url or get_postgres_url()) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE} ({_BASE_TABLE_DDL})")
        cur.execute(
            f"""CREATE TABLE IF NOT EXISTS {FLAGGED_TABLE} (
                {_BASE_TABLE_DDL},
                explanation TEXT,
                explained_at TIMESTAMPTZ
            )"""
        )
        # Forward-compatible migration for databases created by detector v1.
        cur.execute(
            f"ALTER TABLE {PROCESSED_TABLE} "
            "ADD COLUMN IF NOT EXISTS velocity_z_score DOUBLE PRECISION NOT NULL DEFAULT 0"
        )
        cur.execute(
            f"ALTER TABLE {FLAGGED_TABLE} "
            "ADD COLUMN IF NOT EXISTS velocity_z_score DOUBLE PRECISION NOT NULL DEFAULT 0"
        )
        cur.execute(
            f"ALTER TABLE {PROCESSED_TABLE} "
            "ADD COLUMN IF NOT EXISTS detector_version TEXT NOT NULL DEFAULT 'v1'"
        )
        cur.execute(
            f"ALTER TABLE {FLAGGED_TABLE} "
            "ADD COLUMN IF NOT EXISTS detector_version TEXT NOT NULL DEFAULT 'v1'"
        )
        cur.execute(
            f"ALTER TABLE {PROCESSED_TABLE} "
            "ADD COLUMN IF NOT EXISTS evaluation_run_id TEXT NOT NULL DEFAULT 'legacy'"
        )
        cur.execute(
            f"ALTER TABLE {FLAGGED_TABLE} "
            "ADD COLUMN IF NOT EXISTS evaluation_run_id TEXT NOT NULL DEFAULT 'legacy'"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_processed_account_time "
            f"ON {PROCESSED_TABLE} (account_id, event_timestamp DESC)"
        )
        cur.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_processed_transaction_id "
            f"ON {PROCESSED_TABLE} (transaction_id)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_flagged_account_time "
            f"ON {FLAGGED_TABLE} (account_id, event_timestamp DESC)"
        )
        cur.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_flagged_transaction_id "
            f"ON {FLAGGED_TABLE} (transaction_id)"
        )


def write_batch_idempotent(
    batch_df,
    *,
    batch_id: int,
    postgres_url: str,
    jdbc_url: str,
    jdbc_props: dict[str, str],
) -> None:
    """Stage and transactionally merge one Spark micro-batch.

    A repeated ``batch_id`` or a replayed transaction is harmless because both
    destination tables use ``transaction_id`` as their primary key.
    """
    if not isinstance(batch_id, int) or batch_id < 0:
        raise ValueError("batch_id must be a non-negative integer")
    staging_table = f"_fraud_batch_{batch_id}"
    batch_df.write.jdbc(
        url=jdbc_url,
        table=staging_table,
        mode="overwrite",
        properties=jdbc_props,
    )

    column_list = sql.SQL(", ").join(map(sql.Identifier, COLUMNS))
    with psycopg2.connect(postgres_url) as conn, conn.cursor() as cur:
        for destination, predicate in (
            (PROCESSED_TABLE, sql.SQL("TRUE")),
            (FLAGGED_TABLE, sql.SQL("is_flagged = TRUE")),
        ):
            cur.execute(
                sql.SQL(
                    "INSERT INTO {destination} ({columns}) "
                    "SELECT {columns} FROM {staging} WHERE {predicate} "
                    "ON CONFLICT (transaction_id) DO NOTHING"
                ).format(
                    destination=sql.Identifier(destination),
                    columns=column_list,
                    staging=sql.Identifier(staging_table),
                    predicate=predicate,
                )
            )
        cur.execute(sql.SQL("DROP TABLE {staging}").format(staging=sql.Identifier(staging_table)))
