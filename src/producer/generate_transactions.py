"""Continuously publishes synthetic transaction events to the `transactions` Kafka topic.

Events carry an `injected_label` field (null, or one of ANOMALY_TYPES) recording which
anomaly type — if any — this event was deliberately constructed as. This is ground truth
for offline evaluation only: the Spark job in src/streaming must not read this field to
make its flagging decision, only carry it through so flag-vs-label precision/recall can be
computed after the fact.

impossible_travel events support three injection modes (--injection-mode):
  - random (default): pick any other country and let implied speed fall wherever it lands.
    In practice this produces absurdly extreme speeds (tens of thousands of km/h and up),
    since any cross-country jump within the account's typical few-second-to-minute revisit
    gap is astronomically fast almost by construction. Good for a sanity-check "can the rule
    catch anything at all" test, bad for measuring recall near the decision boundary.
  - borderline: back-calculate a country/time-gap combination that targets an implied speed
    of BORDERLINE_SPEED_MIN_KMH-BORDERLINE_SPEED_MAX_KMH — just above
    src/streaming/fraud_detector.py's 900 km/h threshold. Confirmed the rule still achieves
    100% recall here, but that's a property of a hard threshold (any case unambiguously past
    900 gets caught with certainty) more than a statement about calibration.
  - straddle: same back-calculation, but targets STRADDLE_SPEED_MIN_KMH-STRADDLE_SPEED_MAX_KMH
    (700-1100 km/h by default), a band that genuinely straddles the 900 km/h threshold —
    some injected cases should NOT be flagged (below 900) and some should (above 900). This
    is the actual test of boundary behavior; borderline and random both stay safely on one
    side of the threshold and so can't reveal it.

All modes use the same haversine distance math src/streaming/fraud_detector.py uses.
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

load_dotenv()

DEFAULT_TOPIC = "transactions"
DEFAULT_BROKER = os.getenv("KAFKA_BROKER") or "localhost:29092"

COUNTRIES = ["US", "GB", "DE", "FR", "JP", "AU", "BR", "IN", "CA", "SG", "NL", "MX", "ZA", "AE", "KR"]
MERCHANTS = [
    "Amazon", "Walmart", "Target", "Uber", "Starbucks", "Whole Foods", "Shell Gas",
    "Netflix", "Best Buy", "Delta Air Lines", "Marriott", "Apple Store", "Steam",
    "DoorDash", "CVS Pharmacy", "Home Depot", "Costco", "AirBnB", "Spotify", "IKEA",
]

ANOMALY_TYPES = ["impossible_travel", "amount_spike", "velocity_burst"]

# Same country centroids + haversine formula as src/streaming/fraud_detector.py's
# COUNTRY_CENTROIDS / haversine_km — duplicated rather than imported since this module runs
# on the host and that one runs in the Spark container. Keep them in sync by hand.
COUNTRY_CENTROIDS = {
    "US": (39.8, -98.6), "GB": (54.0, -2.0), "DE": (51.2, 10.4), "FR": (46.6, 2.2),
    "JP": (36.2, 138.3), "AU": (-25.3, 133.8), "BR": (-14.2, -51.9), "IN": (20.6, 79.0),
    "CA": (56.1, -106.3), "SG": (1.35, 103.8), "NL": (52.1, 5.3), "MX": (23.6, -102.5),
    "ZA": (-30.6, 22.9), "AE": (23.4, 53.8), "KR": (35.9, 127.8),
}

BORDERLINE_SPEED_MIN_KMH = 1000.0
BORDERLINE_SPEED_MAX_KMH = 3000.0

# Straddles fraud_detector.py's 900 km/h threshold on purpose — unlike BORDERLINE_*, which
# stays safely above it.
STRADDLE_SPEED_MIN_KMH = 700.0
STRADDLE_SPEED_MAX_KMH = 1100.0


def haversine_km(country_a: str, country_b: str) -> float:
    lat1, lon1 = COUNTRY_CENTROIDS[country_a]
    lat2, lon2 = COUNTRY_CENTROIDS[country_b]
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r_km * math.asin(math.sqrt(a))


@dataclass
class Account:
    account_id: str
    home_country: str
    historical_avg: float
    last_location: Optional[str] = None
    last_timestamp: Optional[datetime] = None


def build_accounts(n: int) -> list[Account]:
    # Seeded per account index (not the shared `random` module state) so the same
    # account_id gets the same home_country/historical_avg every time the producer runs —
    # otherwise restarting the producer reshuffles identities out from under any account
    # that survived from a prior run, and re-randomized home_country reads as impossible
    # travel to a detector that (correctly) doesn't know the identity changed underneath it.
    accounts = []
    for i in range(n):
        rng = random.Random(i)
        accounts.append(Account(
            account_id=f"acct_{i:05d}",
            home_country=rng.choice(COUNTRIES),
            historical_avg=round(rng.uniform(30.0, 300.0), 2),
        ))
    return accounts


def next_timestamp(account: Account) -> datetime:
    """Real now, unless that account's last event was fabricated further into the future
    (see borderline_impossible_travel_event) — in which case its logical clock stays ahead
    of wall-clock time rather than snapping back, which would otherwise make the very next
    event look like it arrived before the account's own last transaction."""
    now = datetime.now(timezone.utc)
    return max(now, account.last_timestamp) if account.last_timestamp else now


def make_event(account: Account, amount: float, location: str, label: Optional[str], timestamp: datetime) -> dict:
    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account.account_id,
        "amount": round(amount, 2),
        "merchant": random.choice(MERCHANTS),
        "location": location,
        "timestamp": timestamp.isoformat(),
        "injected_label": label,
    }


def normal_event(account: Account) -> dict:
    amount = max(1.0, random.gauss(account.historical_avg, account.historical_avg * 0.25))
    return make_event(account, amount, account.home_country, label=None, timestamp=next_timestamp(account))


def impossible_travel_event(account: Account) -> dict:
    other_countries = [c for c in COUNTRIES if c != account.home_country]
    foreign_country = random.choice(other_countries)
    amount = max(1.0, random.gauss(account.historical_avg, account.historical_avg * 0.25))
    return make_event(account, amount, foreign_country, label="impossible_travel", timestamp=next_timestamp(account))


def borderline_impossible_travel_event(
    account: Account,
    speed_min_kmh: float = BORDERLINE_SPEED_MIN_KMH,
    speed_max_kmh: float = BORDERLINE_SPEED_MAX_KMH,
) -> Optional[dict]:
    """Back-calculate a country + time-gap that targets an implied travel speed uniformly
    sampled from [speed_min_kmh, speed_max_kmh], using the same haversine math
    fraud_detector.py uses to compute it. Returns None if this account has no prior location
    yet (nothing to compute a speed against — the detector can't flag a first-ever
    transaction either way)."""
    if account.last_location is None:
        return None

    candidates = [c for c in COUNTRIES if c != account.last_location]
    destination = random.choice(candidates)
    distance_km = haversine_km(account.last_location, destination)
    target_speed_kmh = random.uniform(speed_min_kmh, speed_max_kmh)
    elapsed_hours = distance_km / target_speed_kmh

    # Anchored to next_timestamp(account) (>= real now), not account.last_timestamp
    # directly — guarantees this event is never timestamped before the account's actual
    # last event even if next_timestamp already had to advance the clock past real "now"
    # for a prior borderline injection.
    timestamp = next_timestamp(account) + timedelta(hours=elapsed_hours)
    amount = max(1.0, random.gauss(account.historical_avg, account.historical_avg * 0.25))
    return make_event(account, amount, destination, label="impossible_travel", timestamp=timestamp)


def amount_spike_event(account: Account) -> dict:
    amount = account.historical_avg * random.uniform(8, 25)
    return make_event(account, amount, account.home_country, label="amount_spike", timestamp=next_timestamp(account))


def velocity_burst_events(account: Account) -> list[dict]:
    burst_size = random.randint(4, 8)
    events = []
    for _ in range(burst_size):
        amount = max(1.0, random.gauss(account.historical_avg, account.historical_avg * 0.15))
        events.append(make_event(account, amount, account.home_country, label="velocity_burst", timestamp=next_timestamp(account)))
    return events


def touch_account(account: Account, event: dict) -> None:
    account.last_location = event["location"]
    account.last_timestamp = datetime.fromisoformat(event["timestamp"])


def build_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=10,
    )


def run(
    rate: float, broker: str, topic: str, num_accounts: int, anomaly_rate: float,
    duration: Optional[float], injection_mode: str = "random",
) -> None:
    accounts = build_accounts(num_accounts)

    try:
        producer = build_producer(broker)
    except NoBrokersAvailable:
        raise SystemExit(
            f"Could not reach Kafka at {broker}. Is it running? Try: docker compose up -d kafka"
        )

    sleep_interval = 1.0 / rate if rate > 0 else 0
    start = time.monotonic()
    sent = 0
    anomaly_counts = {t: 0 for t in ANOMALY_TYPES}

    print(f"Publishing to topic '{topic}' on {broker} at ~{rate} events/sec (Ctrl+C to stop)")

    try:
        while duration is None or (time.monotonic() - start) < duration:
            account = random.choice(accounts)
            is_anomalous = random.random() < anomaly_rate

            if not is_anomalous:
                events = [normal_event(account)]
                label = None
            else:
                anomaly_type = random.choice(ANOMALY_TYPES)
                label = anomaly_type
                if anomaly_type == "impossible_travel":
                    if injection_mode == "borderline":
                        event = borderline_impossible_travel_event(account)
                    elif injection_mode == "straddle":
                        event = borderline_impossible_travel_event(
                            account, speed_min_kmh=STRADDLE_SPEED_MIN_KMH, speed_max_kmh=STRADDLE_SPEED_MAX_KMH,
                        )
                    else:
                        event = impossible_travel_event(account)
                    if event is None:
                        # No prior location for this account yet (borderline mode needs one
                        # to compute a distance/speed against) — nothing to inject this
                        # round, fall back to a normal transaction.
                        events = [normal_event(account)]
                        label = None
                    else:
                        events = [event]
                elif anomaly_type == "amount_spike":
                    events = [amount_spike_event(account)]
                else:
                    events = velocity_burst_events(account)

            for event in events:
                producer.send(topic, key=event["account_id"], value=event)
                touch_account(account, event)
                sent += 1

            if label:
                anomaly_counts[label] += 1
                print(f"  [anomaly:{label}] {account.account_id} x{len(events)} event(s)")

            if sent % 50 < len(events):
                print(f"sent={sent} anomalies={anomaly_counts}")

            time.sleep(sleep_interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        producer.flush()
        producer.close()
        print(f"Done. sent={sent} anomalies={anomaly_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a live transaction stream into Kafka.")
    parser.add_argument("--rate", type=float, default=5.0, help="events per second (default: 5.0)")
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER, help="Kafka bootstrap broker, host:port")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Kafka topic to publish to")
    parser.add_argument(
        "--accounts", type=int, default=500,
        help=(
            "number of synthetic accounts to simulate (default: 500). Keep this large "
            "relative to --rate: with too few accounts, random chance alone puts many "
            "transactions per account in any 5-minute window, drowning out the velocity "
            "rule's burst signal (e.g. 50 accounts @ 8/sec puts ~14 in-window by pure "
            "chance, swamping a 4-8 event injected burst)."
        ),
    )
    parser.add_argument("--anomaly-rate", type=float, default=0.05, help="fraction of events that are anomalous")
    parser.add_argument("--duration", type=float, default=None, help="seconds to run before stopping (default: forever)")
    parser.add_argument(
        "--injection-mode", type=str, choices=["random", "borderline", "straddle"], default="random",
        help=(
            "how impossible_travel anomalies are constructed (default: random). 'random' "
            "picks any other country and lets implied speed fall wherever — usually "
            f"absurdly extreme. 'borderline' back-calculates a country/time-gap targeting "
            f"{BORDERLINE_SPEED_MIN_KMH:.0f}-{BORDERLINE_SPEED_MAX_KMH:.0f} km/h, just above "
            "the detector's 900 km/h threshold. 'straddle' targets "
            f"{STRADDLE_SPEED_MIN_KMH:.0f}-{STRADDLE_SPEED_MAX_KMH:.0f} km/h, genuinely "
            "crossing the threshold — the real test of boundary behavior, since some "
            "injected cases should be flagged and some should not."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        rate=args.rate,
        broker=args.broker,
        topic=args.topic,
        num_accounts=args.accounts,
        anomaly_rate=args.anomaly_rate,
        duration=args.duration,
        injection_mode=args.injection_mode,
    )
