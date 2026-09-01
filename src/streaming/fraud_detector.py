"""Spark Structured Streaming job: consumes `transactions` from Kafka, maintains rolling
per-account state (recent transaction timestamps, last location, running amount/velocity
averages), and flags transactions that trip one or more rules:

  - velocity:  the account's live 5-minute sliding transaction count is more than two
               standard deviations above its own historical baseline — the count seen in
               its past *completed* 5-minute tumbling windows only.
               The baseline deliberately excludes the in-progress window: folding a burst's
               own events into its own baseline as they happen would let the anomaly chase
               (and dilute) the very average it's being compared against.
  - amount:    amount exceeds AMOUNT_SPIKE_MULTIPLIER x the account's running historical
               average (once enough observations exist to trust that average)
  - geo:       the implied speed between this transaction's location and the account's
               previous one exceeds IMPOSSIBLE_SPEED_KMH (faster than commercial air travel)

The rule engine only ever looks at fields a real detector would have (location, amount,
timestamp, account_id) — never at `injected_label`, which the producer attaches purely as
ground truth for offline evaluation. Every processed transaction is written to
processed_transactions; transactions that tripped >=1 rule are additionally written to
flagged_transactions, both including injected_label so precision/recall per rule can be
computed against it after the fact.

Run inside the spark container:
    docker compose exec spark spark-submit \\
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9,org.postgresql:postgresql:42.7.3 \\
        /opt/spark-apps/streaming/fraud_detector.py
"""

import json
import os
import time

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.detection.rules import amount_signal, geo_signal, velocity_signal
from src.storage import db_writer
from src.streaming.watchdog import ProgressWatchdog, StreamStalledError

KAFKA_BROKER = os.getenv("KAFKA_BROKER_INTERNAL", "kafka:9092")
KAFKA_TOPIC = "transactions"
DETECTOR_VERSION = "v2"
CHECKPOINT_LOCATION = os.getenv(
    "CHECKPOINT_LOCATION", "/opt/fraud-pipeline/checkpoints/fraud_detector_v2_2"
)
STREAM_STALL_TIMEOUT_SECONDS = float(os.getenv("STREAM_STALL_TIMEOUT_SECONDS", "180"))
STREAM_RESTART_LIMIT = int(os.getenv("STREAM_RESTART_LIMIT", "5"))
STREAM_RESTART_BACKOFF_SECONDS = float(os.getenv("STREAM_RESTART_BACKOFF_SECONDS", "10"))
MAX_OFFSETS_PER_TRIGGER = int(os.getenv("MAX_OFFSETS_PER_TRIGGER", "1000"))
STREAM_TRIGGER_SECONDS = float(os.getenv("STREAM_TRIGGER_SECONDS", "5"))

VELOCITY_WINDOW_MS = 5 * 60 * 1000
# Rule thresholds live in src.detection.rules so the Spark adapter and unit tests share
# one source of truth.

INPUT_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("account_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("merchant", StringType()),
    StructField("location", StringType()),
    StructField("timestamp", StringType()),
    StructField("injected_label", StringType()),
    StructField("evaluation_run_id", StringType()),
])

OUTPUT_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("detector_version", StringType()),
    StructField("evaluation_run_id", StringType()),
    StructField("account_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("merchant", StringType()),
    StructField("location", StringType()),
    StructField("event_timestamp", TimestampType()),
    StructField("injected_label", StringType()),
    StructField("flag_velocity", BooleanType()),
    StructField("flag_amount", BooleanType()),
    StructField("flag_geo", BooleanType()),
    StructField("is_flagged", BooleanType()),
    StructField("triggered_rules", StringType()),
    StructField("velocity_count_5min", IntegerType()),
    StructField("velocity_ratio_to_baseline", DoubleType()),
    StructField("velocity_z_score", DoubleType()),
    StructField("amount_ratio_to_avg", DoubleType()),
    StructField("implied_speed_kmh", DoubleType()),
])

STATE_SCHEMA = StructType([
    StructField("last_location", StringType()),
    StructField("last_event_time_ms", LongType()),
    StructField("amount_count", LongType()),
    StructField("amount_mean", DoubleType()),
    StructField("recent_timestamps_json", StringType()),
    StructField("velocity_window_start_ms", LongType()),
    StructField("velocity_window_count", LongType()),
    StructField("velocity_baseline_count", LongType()),
    StructField("velocity_baseline_mean", DoubleType()),
    StructField("velocity_baseline_m2", DoubleType()),
])


def update_account_state(key, pdf_iter, state: GroupState):
    if state.exists:
        (last_location, last_event_time_ms, amount_count, amount_mean, recent_ts_json,
         velocity_window_start_ms, velocity_window_count,
         velocity_baseline_count, velocity_baseline_mean,
         velocity_baseline_m2) = state.get
        recent_timestamps = json.loads(recent_ts_json) if recent_ts_json else []
    else:
        last_location, last_event_time_ms = None, None
        amount_count, amount_mean = 0, 0.0
        recent_timestamps = []
        velocity_window_start_ms, velocity_window_count = None, 0
        velocity_baseline_count, velocity_baseline_mean, velocity_baseline_m2 = 0, 0.0, 0.0

    out_rows = []
    for pdf in pdf_iter:
        for row in pdf.sort_values("event_timestamp").itertuples(index=False):
            event_time_ms = int(row.event_timestamp.timestamp() * 1000)

            recent_timestamps = [t for t in recent_timestamps if event_time_ms - t <= VELOCITY_WINDOW_MS]
            velocity_count = len(recent_timestamps) + 1

            # Baseline uses only *completed* tumbling windows, finalized before this event's
            # own (still in-progress) window — so an ongoing burst can never pollute the
            # baseline it's being compared against.
            if velocity_window_start_ms is None:
                velocity_window_start_ms = event_time_ms
            elif event_time_ms - velocity_window_start_ms >= VELOCITY_WINDOW_MS:
                new_baseline_count = velocity_baseline_count + 1
                delta = velocity_window_count - velocity_baseline_mean
                velocity_baseline_mean += delta / new_baseline_count
                delta_after_update = velocity_window_count - velocity_baseline_mean
                velocity_baseline_m2 += delta * delta_after_update
                velocity_baseline_count = new_baseline_count
                velocity_window_start_ms = event_time_ms
                velocity_window_count = 0

            flag_velocity, velocity_ratio, velocity_z_score = velocity_signal(
                velocity_count,
                velocity_baseline_count,
                velocity_baseline_mean,
                velocity_baseline_m2,
            )

            elapsed_ms = event_time_ms - last_event_time_ms if last_event_time_ms else None
            flag_geo, implied_speed_kmh = geo_signal(last_location, row.location, elapsed_ms)
            flag_amount, amount_ratio = amount_signal(row.amount, amount_count, amount_mean)

            triggered = [
                name for name, fired in (
                    ("velocity", flag_velocity),
                    ("amount", flag_amount),
                    ("geo", flag_geo),
                ) if fired
            ]

            out_rows.append({
                "transaction_id": row.transaction_id,
                "detector_version": DETECTOR_VERSION,
                "evaluation_run_id": row.evaluation_run_id,
                "account_id": row.account_id,
                "amount": row.amount,
                "merchant": row.merchant,
                "location": row.location,
                "event_timestamp": row.event_timestamp,
                "injected_label": row.injected_label,
                "flag_velocity": flag_velocity,
                "flag_amount": flag_amount,
                "flag_geo": flag_geo,
                "is_flagged": len(triggered) > 0,
                "triggered_rules": ",".join(triggered),
                "velocity_count_5min": velocity_count,
                # Persist full precision so offline threshold sweeps reproduce the
                # detector's strict boundaries exactly; presentation layers format it.
                "velocity_ratio_to_baseline": velocity_ratio,
                "velocity_z_score": velocity_z_score,
                "amount_ratio_to_avg": amount_ratio,
                "implied_speed_kmh": implied_speed_kmh,
            })

            recent_timestamps.append(event_time_ms)
            new_amount_count = amount_count + 1
            amount_mean = amount_mean + (row.amount - amount_mean) / new_amount_count
            amount_count = new_amount_count
            velocity_window_count += 1
            last_location = row.location
            last_event_time_ms = event_time_ms

    state.update((
        last_location, last_event_time_ms, amount_count, amount_mean, json.dumps(recent_timestamps),
        velocity_window_start_ms, velocity_window_count,
        velocity_baseline_count, velocity_baseline_mean, velocity_baseline_m2,
    ))

    columns = [f.name for f in OUTPUT_SCHEMA.fields]
    yield pd.DataFrame(out_rows, columns=columns)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("fraud-detector")
        .config("spark.sql.shuffle.partitions", "4")
        # AdminClient-based offset fetching (Spark's default since 3.x) has been observed to
        # hang indefinitely mid-stream against some Kafka broker setups; the older
        # consumer-based offset fetch path is slower per-call but doesn't exhibit the hang.
        .config("spark.sql.streaming.kafka.useDeprecatedOffsetFetching", "true")
        .getOrCreate()
    )


def start_query(spark: SparkSession):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", str(MAX_OFFSETS_PER_TRIGGER))
        .load()
    )

    parsed = (
        raw.select(from_json(col("value").cast("string"), INPUT_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_timestamp", to_timestamp(col("timestamp")))
        .drop("timestamp")
    )

    flagged_and_processed = parsed.groupBy("account_id").applyInPandasWithState(
        update_account_state,
        OUTPUT_SCHEMA,
        STATE_SCHEMA,
        "Update",
        GroupStateTimeout.NoTimeout,
    )

    postgres_url = db_writer.get_postgres_url()
    jdbc_url, jdbc_props = db_writer.jdbc_config(postgres_url)
    db_writer.ensure_schema(postgres_url)

    def write_batch(batch_df, batch_id):
        batch_df.persist()
        try:
            if not batch_df.isEmpty():
                db_writer.write_batch_idempotent(
                    batch_df,
                    batch_id=batch_id,
                    postgres_url=postgres_url,
                    jdbc_url=jdbc_url,
                    jdbc_props=jdbc_props,
                )
        finally:
            batch_df.unpersist()

    return (
        flagged_and_processed.writeStream
        .outputMode("update")
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_LOCATION)
        .trigger(processingTime=f"{STREAM_TRIGGER_SECONDS:g} seconds")
        .start()
    )


def await_with_watchdog(query) -> None:
    watchdog = ProgressWatchdog(timeout_seconds=STREAM_STALL_TIMEOUT_SECONDS)
    while query.isActive:
        progress = query.lastProgress
        token = progress.get("timestamp") if progress else None
        if watchdog.observe(token, time.monotonic()):
            query.stop()
            raise StreamStalledError(
                f"No Spark progress heartbeat for {STREAM_STALL_TIMEOUT_SECONDS:.0f} seconds"
            )
        query.awaitTermination(10)


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    for attempt in range(STREAM_RESTART_LIMIT + 1):
        query = start_query(spark)
        try:
            await_with_watchdog(query)
            return
        except StreamStalledError:
            if attempt >= STREAM_RESTART_LIMIT:
                raise
            wait = STREAM_RESTART_BACKOFF_SECONDS * (attempt + 1)
            print(f"Stream stalled; restarting from checkpoint in {wait:.0f}s", flush=True)
            time.sleep(wait)


if __name__ == "__main__":
    main()
