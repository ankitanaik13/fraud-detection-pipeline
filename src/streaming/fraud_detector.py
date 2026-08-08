"""Spark Structured Streaming job: consumes `transactions` from Kafka, maintains rolling
per-account state (recent transaction timestamps, last location, running amount/velocity
averages), and flags transactions that trip one or more rules:

  - velocity:  the account's live 5-minute sliding transaction count exceeds
               VELOCITY_SPIKE_MULTIPLIER x that account's own historical baseline — the
               average count seen in its past *completed* 5-minute tumbling windows only.
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
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # /opt/spark-apps, for `storage.db_writer`

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

from storage import db_writer

KAFKA_BROKER = os.getenv("KAFKA_BROKER_INTERNAL", "kafka:9092")
KAFKA_TOPIC = "transactions"
CHECKPOINT_LOCATION = "/opt/spark-apps/streaming/checkpoints/fraud_detector"

VELOCITY_WINDOW_MS = 5 * 60 * 1000
# Each observation here is one *completed* 5-minute window, not one transaction — a
# freshly-deployed detector (or a short local demo run) may only accumulate a handful of
# completed windows per account, so this gate has to be low enough to actually open in that
# time. A long-running production deployment would naturally build a deeper baseline.
VELOCITY_MIN_OBSERVATIONS = 1
# Empirically swept against a 20-min/10k-event soak run: at this account pool size and
# burst size (4-8 events), the velocity signal is only modestly separable from an account's
# own baseline (bursts average ~1.5x baseline vs ~1.05x for normal windows) — no threshold
# achieves both good precision and good recall here. 1.6 is near the best precision/recall
# balance point found (~26%/~28%); this is a real property of the injected signal size
# relative to typical per-account window activity, not a rule-logic bug.
VELOCITY_SPIKE_MULTIPLIER = 1.6
AMOUNT_MIN_OBSERVATIONS = 3
AMOUNT_SPIKE_MULTIPLIER = 5.0
IMPOSSIBLE_SPEED_KMH = 900.0

# Rough centroid (lat, lon) per country code used by the producer — good enough to tell
# "clearly impossible" travel apart from "plausible", not a real geolocation source.
COUNTRY_CENTROIDS = {
    "US": (39.8, -98.6), "GB": (54.0, -2.0), "DE": (51.2, 10.4), "FR": (46.6, 2.2),
    "JP": (36.2, 138.3), "AU": (-25.3, 133.8), "BR": (-14.2, -51.9), "IN": (20.6, 79.0),
    "CA": (56.1, -106.3), "SG": (1.35, 103.8), "NL": (52.1, 5.3), "MX": (23.6, -102.5),
    "ZA": (-30.6, 22.9), "AE": (23.4, 53.8), "KR": (35.9, 127.8),
}

INPUT_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("account_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("merchant", StringType()),
    StructField("location", StringType()),
    StructField("timestamp", StringType()),
    StructField("injected_label", StringType()),
])

OUTPUT_SCHEMA = StructType([
    StructField("transaction_id", StringType()),
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
])


def haversine_km(country_a: str, country_b: str) -> float:
    if country_a not in COUNTRY_CENTROIDS or country_b not in COUNTRY_CENTROIDS:
        return 0.0
    lat1, lon1 = COUNTRY_CENTROIDS[country_a]
    lat2, lon2 = COUNTRY_CENTROIDS[country_b]
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r_km * math.asin(math.sqrt(a))


def update_account_state(key, pdf_iter, state: GroupState):
    if state.exists:
        (last_location, last_event_time_ms, amount_count, amount_mean, recent_ts_json,
         velocity_window_start_ms, velocity_window_count,
         velocity_baseline_count, velocity_baseline_mean) = state.get
        recent_timestamps = json.loads(recent_ts_json) if recent_ts_json else []
    else:
        last_location, last_event_time_ms = None, None
        amount_count, amount_mean = 0, 0.0
        recent_timestamps = []
        velocity_window_start_ms, velocity_window_count = None, 0
        velocity_baseline_count, velocity_baseline_mean = 0, 0.0

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
                velocity_baseline_mean += (velocity_window_count - velocity_baseline_mean) / new_baseline_count
                velocity_baseline_count = new_baseline_count
                velocity_window_start_ms = event_time_ms
                velocity_window_count = 0

            velocity_ratio = (velocity_count / velocity_baseline_mean) if velocity_baseline_mean > 0 else 0.0
            flag_velocity = (
                velocity_baseline_count >= VELOCITY_MIN_OBSERVATIONS
                and velocity_count > velocity_baseline_mean * VELOCITY_SPIKE_MULTIPLIER
            )

            implied_speed_kmh = 0.0
            flag_geo = False
            if last_location is not None and row.location != last_location:
                distance_km = haversine_km(last_location, row.location)
                hours_elapsed = max((event_time_ms - last_event_time_ms) / 3_600_000.0, 1e-6)
                implied_speed_kmh = distance_km / hours_elapsed
                flag_geo = implied_speed_kmh > IMPOSSIBLE_SPEED_KMH

            amount_ratio = (row.amount / amount_mean) if amount_mean > 0 else 0.0
            flag_amount = amount_count >= AMOUNT_MIN_OBSERVATIONS and row.amount > amount_mean * AMOUNT_SPIKE_MULTIPLIER

            triggered = [
                name for name, fired in (
                    ("velocity", flag_velocity),
                    ("amount", flag_amount),
                    ("geo", flag_geo),
                ) if fired
            ]

            out_rows.append({
                "transaction_id": row.transaction_id,
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
                "velocity_ratio_to_baseline": round(velocity_ratio, 3),
                "amount_ratio_to_avg": round(amount_ratio, 3),
                "implied_speed_kmh": round(implied_speed_kmh, 1),
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
        velocity_baseline_count, velocity_baseline_mean,
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


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", "200")
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

    jdbc_url, jdbc_props = db_writer.jdbc_config()

    def write_batch(batch_df, batch_id):
        batch_df.persist()
        try:
            if not batch_df.isEmpty():
                db_writer.write_processed_batch(batch_df, jdbc_url, jdbc_props)
                db_writer.write_flagged_batch(batch_df, jdbc_url, jdbc_props)
        finally:
            batch_df.unpersist()

    query = (
        flagged_and_processed.writeStream
        .outputMode("update")
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_LOCATION)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
