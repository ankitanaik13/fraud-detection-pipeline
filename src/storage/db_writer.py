"""JDBC writers used by src/streaming/fraud_detector.py's foreachBatch to persist processed
+ flagged transactions to Postgres. Table creation is handled by Spark's JDBC writer on the
first append to a table that doesn't exist yet, inferring column types from the DataFrame.
"""

import os
from urllib.parse import urlparse

PROCESSED_TABLE = "processed_transactions"
FLAGGED_TABLE = "flagged_transactions"

DEFAULT_POSTGRES_URL = "postgresql://fraud:fraud@postgres:5432/frauddb"


def jdbc_config(postgres_url: str = None) -> tuple[str, dict]:
    """Convert a postgresql:// URL (as used by POSTGRES_URL) into a Spark JDBC url + properties."""
    postgres_url = postgres_url or os.getenv("POSTGRES_URL") or DEFAULT_POSTGRES_URL
    parsed = urlparse(postgres_url)
    jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
    properties = {
        "user": parsed.username or "fraud",
        "password": parsed.password or "fraud",
        "driver": "org.postgresql.Driver",
    }
    return jdbc_url, properties


def write_processed_batch(batch_df, jdbc_url: str, jdbc_props: dict) -> None:
    batch_df.write.jdbc(url=jdbc_url, table=PROCESSED_TABLE, mode="append", properties=jdbc_props)


def write_flagged_batch(batch_df, jdbc_url: str, jdbc_props: dict) -> None:
    flagged_df = batch_df.filter(batch_df.is_flagged)
    flagged_df.write.jdbc(url=jdbc_url, table=FLAGGED_TABLE, mode="append", properties=jdbc_props)
