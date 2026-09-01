# Real-Time Transaction Fraud Detection Pipeline

[![CI](https://github.com/ankitanaik13/fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitanaik13/fraud-detection-pipeline/actions/workflows/ci.yml)

A local, four-stage pipeline that simulates a live transaction stream, flags anomalies with
per-account stateful rules in Spark, explains each flag in plain English via an LLM, and
exposes account risk status as an MCP tool. Runs entirely on Docker Compose — no GPU, no
notebooks, no cloud dependency except the LLM call itself.

Built as a portfolio project to demonstrate the full loop: synthetic ground truth →
streaming detection → measured precision/recall → tuning against real numbers → LLM
enrichment → a queryable tool interface. Detector versions and evaluation-run IDs are
persisted with every prediction, so benchmark reports cannot silently mix algorithms,
cold-start traffic, or earlier experiments.

## Architecture

```
generate_transactions.py  (producer)
        │  JSON events, keyed by account_id, ~5% deliberately anomalous
        ▼
   Kafka topic: transactions
        │
        ▼
fraud_detector.py  (Spark Structured Streaming, applyInPandasWithState)
  per-account rolling state → amount / geo / velocity rules
        │
        ├────────────▶ Postgres: processed_transactions   (every transaction)
        └────────────▶ Postgres: flagged_transactions      (rule(s) tripped)
                        ▲                    │
                        │ writes .explanation │ reads unexplained rows
                        │                     ▼
                        └───── llm_explainer.py  (NVIDIA NIM, meta/llama-3.1-70b-instruct)

              risk_status_server.py  (MCP server, stdio)
                  reads processed_transactions + flagged_transactions
                                │
                                ▼
              MCP client  →  get_account_risk_status(account_id)
```

Every row in both Postgres tables carries the producer's `injected_label` (ground truth —
which anomaly, if any, was deliberately injected) alongside the detector's own flags. The
rule engine never reads that field; it exists purely so precision/recall per rule can be
computed after the fact.

## Quickstart

```bash
cp .env.example .env
docker compose up -d --build
python -m src.producer.generate_transactions --rate 8

# once some transactions are flagged
python -m src.explain.llm_explainer --limit 50
python -m src.mcp.risk_status_server   # stdio transport
```

Compose waits for Kafka and Postgres health checks, then starts the Spark job
automatically. The checkpoint is stored in a named volume, so a container restart resumes
from committed offsets instead of replaying the topic from scratch.

`generate_transactions.py --injection-mode {random,borderline,straddle}` controls how
`impossible_travel` anomalies are constructed — see "The geo finding" below for why this
matters and what each mode is for.

## Detection rules (`src/streaming/fraud_detector.py`)

Per-account rolling state, maintained via `applyInPandasWithState`:

| Rule | Trips when | Needs |
|---|---|---|
| **amount** | amount > 5x the account's running historical average | ≥3 prior transactions |
| **geo** | implied travel speed between consecutive transaction locations > 900 km/h | a previous transaction to compare against |
| **velocity** | live 5-minute count >2 standard deviations above the account's completed-window baseline, plus a 1.2x ratio guard | ≥5 completed windows |

Velocity v2 maintains mean and variance online with Welford's algorithm. Its effective
standard deviation is conservatively floored at the Poisson expectation `sqrt(mean)`, so
a few identical low-count windows cannot create an unstable or infinite z-score. The
baseline excludes the in-progress window—otherwise a burst would dilute its own signal.

## Current v2 release result

The quality-gated run `v2-release-20260901` processed 3,427 events: five normal history
windows for each of 500 run-isolated accounts, followed by a 75-second straddle benchmark.
The gate required at least 20 positive examples per rule and at least 0.50 precision and
recall; all three rules passed.

| Rule | Positive examples | Precision | Recall | F1 | Precision 95% CI |
|---|---:|---:|---:|---:|---:|
| amount | 63 | 100.0% | 98.4% | 99.2% | 94.2–100% |
| geo | 33 | 100.0% | 100.0% | 100.0% | 89.6–100% |
| velocity | 419 | 89.9% | 61.6% | 73.1% | 85.9–92.9% |
| overall | 515 | 92.9% | 68.5% | 78.9% | 89.9–95.1% |

The report includes confusion matrices, Wilson intervals, the exact rule configuration,
and a velocity threshold sweep. See [`outputs/v2_release_metrics.json`](outputs/v2_release_metrics.json).
Risk scoring uses each rule's **95% precision lower bound**, not the point estimate.

## Historical v1 benchmark (20-minute soak, 500 accounts, 10,028 events)

These figures describe the retired ratio-to-mean velocity detector (`v1`), not the current
variance-aware detector. They are retained as an engineering audit trail and as the reason
v2 was built; they must not be presented as v2 performance.

| Rule | Precision | Recall |
|---|---|---|
| amount | 100% | 77% |
| geo | 50% | 94% |
| velocity | 26% | 28% |
| overall (any rule) | 38% | 43% |

### Why velocity v1 was replaced

Recall is high across the board (77-94%) except velocity. This isn't a tuning miss — it's
swept:

| Threshold (×baseline) | Precision | Recall |
|---|---|---|
| 1.2 | 20% | 46% |
| 1.4 | 23% | 36% |
| **1.6 (chosen)** | **26%** | **28%** |
| 1.8 | 28% | 21% |
| 2.0 | 27% | 12% |
| 2.5 | 20% | 5% |
| 3.0 | 15% | 2% |

No v1 ratio threshold gets both good precision and good recall—precision tops out around 26-28%
no matter where the line is drawn. **Root cause:** a 4-8 event injected burst only pushes an
account's 5-minute window count to ~1.5x its own baseline on average (max observed 5x);
normal (non-burst) windows already sit at ~1.0-1.1x baseline just from ordinary variance.
The signal and the noise floor are too close together for a ratio-to-mean test to separate
them cleanly. v2 implements the variance-aware statistical test identified by that audit.
Its performance is published only after the versioned quality gate has sufficient samples.

### The geo finding: is 94% recall real, or trivial?

The initial 94% geo recall was measured against `--injection-mode random` (the default) —
which picks any other country and lets implied speed fall wherever. Checking the actual
distribution of injected speeds revealed why that number wasn't meaningful on its own:
median 539,632 km/h, minimum (excluding first-transaction edge cases) 13,703 km/h — every
injected case was 15x to 100,000x+ past the 900 km/h threshold. Recall on cases that
extreme says nothing about calibration near the actual decision boundary.

Two follow-up injection modes make this testable:

- `--injection-mode borderline`: targets 1,000-3,000 km/h (just above threshold, using the
  same haversine math as the detector) — 148/148 non-edge-case injections still caught, 100%
  recall. But this range is still safely on one side of 900, so 100% recall here is
  partly *guaranteed* by construction — a hard threshold can't miss a case unambiguously
  past it.
- `--injection-mode straddle`: targets 700-1,100 km/h, **genuinely crossing** 900 — some
  injected cases should be flagged, some shouldn't. This is the real test. Result, scored
  against each case's actual achieved speed (not just its label):

  | | n | Correct | Incorrect |
  |---|---|---|---|
  | Below 900 (should NOT flag) | 80 | 80 unflagged | 0 flagged |
  | Above 900 (should flag) | 60 | 60 caught | 0 missed |

  **100% precision, 100% recall, zero errors either direction**, across an achieved-speed
  range of 699.6-1,091.2 km/h. The rule's boundary behavior is exact, not fuzzy — as
  expected for a hard `>900` threshold, but worth confirming empirically since it also
  validates the producer's back-calculation and the detector's canonical shared haversine
  implementation agree with each other precisely.

  Scoring this same run against `injected_label` alone (the usual methodology) gives a
  misleading TP=60/FP=131/FN=80 — "43% recall." That's a scoring-convention artifact, not a
  rule failure: 80 of those "misses" are cases deliberately constructed below threshold,
  which the rule is *correct* to leave unflagged. The 131 FPs are a separate, already-known
  effect: the transaction immediately *after* an injected geo event — the account's next
  transaction back home — itself reads as a huge implied speed (near-zero elapsed time since
  the injected event), so it gets flagged too even though it isn't the labeled anomaly. A
  labeling artifact of chaining two real events, not a detector bug.

## Explanations (`src/explain/llm_explainer.py`)

For each flagged transaction, calls NVIDIA NIM (`meta/llama-3.1-70b-instruct`) to generate a
one-sentence plain-English explanation grounded strictly in the rule(s) that fired and that
transaction's specific values — never raw guesswork. Confidence language is calibrated to
each rule's measured precision above: stated directly for amount, hedged for geo, hedged
strongly (and flagged as often-a-false-positive) for velocity. Writes back to
`flagged_transactions.explanation`; safe to re-run, only unexplained rows are processed.

```bash
python -m src.explain.llm_explainer --limit 50
python -m src.explain.llm_explainer --transaction-ids id1,id2
```

Untrusted transaction fields are bounded and isolated inside explicit prompt-data tags.
Generated text is accepted only when it is a single, non-Markdown sentence under 600
characters; invalid generations are retried rather than written to the database.

**Two real bugs surfaced during testing:**

1. **Country-code hallucination.** The model initially read location code `DE` as the US
   state Delaware instead of Germany. Fixed by spelling out full country names in the
   prompt (`DE (Germany)`) instead of leaving 2-letter codes for the model to interpret —
   removes the ambiguity instead of hoping the model resolves it correctly.
2. **Unbounded hang on the NIM call.** The `OpenAI` client had no request timeout, so the
   first call in a session could hang indefinitely with no exception raised — nothing to
   retry, just silence. Fixed with an explicit 30s client timeout, turning hangs into
   retryable `APITimeoutError`s. The free tier has been consistently slow enough during
   development that this fires on a large fraction of calls; the retry/backoff logic is
   handling real, not just theoretical, failures.

## Risk status MCP server (`src/mcp/risk_status_server.py`)

Exposes one tool, `get_account_risk_status(account_id)`, over stdio via the official MCP
Python SDK. Queries Postgres for that account's flagged-transaction history and returns:

- `risk_score` — sum of conservative v2 precision lower bounds across the account's v2
  flags (amount 0.9417, geo 0.8957, velocity 0.8586); the response explicitly returns
  `scoring_calibration: "v2-release-20260901-precision-ci95-lower"`
- `risk_level` — `none` / `low` / `medium` / `high`, thresholded on `risk_score`
- `flags_by_rule`, `total_flags`, `total_processed_transactions`
- `recent_flags` — up to the last 20 flagged transactions, each with its triggered rule(s),
  injected label (ground truth, for dev/testing), and LLM explanation where generated

```bash
python -m src.mcp.risk_status_server   # stdio transport
```

The MCP boundary validates account IDs before querying Postgres, and all queries remain
parameterized.

## Delivery semantics and stall recovery

- **Idempotent sink:** each micro-batch is written to a staging table and merged into
  Postgres on the stable `transaction_id` primary key. If Spark retries after a database
  commit but before checkpoint advancement, the replay cannot duplicate transactions.
- **Durable checkpoint:** Kafka offsets and state-store progress live in a named Docker
  volume.
- **Progress watchdog:** a ten-second processing trigger creates a regular progress
  heartbeat. If no new heartbeat appears for `STREAM_STALL_TIMEOUT_SECONDS`, the query is
  stopped and restarted from its checkpoint with bounded backoff.
- **Schema ownership:** the sink creates explicit tables, primary keys, constraints, and
  account/time indexes rather than relying on Spark's inferred schema.

The watchdog turns the previously silent Kafka/Spark stall into an observable and
recoverable event. It mitigates the connector behavior; it does not claim to fix the
upstream connector itself.

## Reproducible evaluation

The checked-in evaluator derives rule-level and overall confusion matrices, precision,
recall, F1, and Wilson 95% confidence intervals directly from
`processed_transactions`. First run a benchmark with normal history preloaded for each
account; otherwise a short test measures cold-start behavior rather than steady state:

```bash
RUN_ID="v2-straddle-$(date -u +%Y%m%dT%H%M%SZ)"
python -m src.producer.generate_transactions \
  --rate 8 --accounts 500 --anomaly-rate 0.12 --duration 180 \
  --warmup-windows 5 --injection-mode straddle --run-id "$RUN_ID"

# after Spark consumes the run
python -m src.evaluation.evaluate_database \
  --detector-version v2 --run-id "$RUN_ID" --fail-on-quality-gate
```

It scores geo straddle cases against achieved speed, not merely the injected label, so
deliberately below-threshold examples are not misreported as false negatives—either at
the geo-rule level or in overall metrics. The default release gate requires at least 20
positive examples per rule and at least 0.50 precision and recall. Reports include their
detector version, run ID, generation time, sample counts, metrics, intervals, and explicit
gate failures; the default output is `outputs/latest_metrics.json`.

The producer derives an account-ID namespace from each run ID, preventing Spark state from
one experiment contaminating another. Travel injections also update the simulated account's
current location, so the next normal event is not an unlabeled impossible return trip.

## Engineering checks

```bash
pip install -r requirements-dev.txt
make check
docker compose config --quiet
```

The unit suite covers strict rule boundaries, variance-aware velocity behavior, zero
baselines, shared geospatial math, precision-weighted risk scoring, straddle and overall
evaluation semantics, confidence intervals, quality gates, watchdog behavior,
idempotent sink construction, URL validation, and LLM output guards. GitHub Actions runs
linting, 85% coverage enforcement, Compose validation, and the Spark image build.

### Historical v1 sample output

This payload is retained only to show the MCP response shape from the earlier soak run;
its score and explanation wording use the retired v1 calibration. Current v2 responses
also include `detector_version` and `scoring_calibration` and use the release weights above.

`get_account_risk_status("acct_00434")` — an account with all three rule types represented:

```json
{
  "account_id": "acct_00434",
  "risk_score": 4.34,
  "risk_level": "high",
  "total_flags": 11,
  "total_processed_transactions": 22,
  "flags_by_rule": { "amount": 1, "geo": 2, "velocity": 9 },
  "last_explanation": "This transaction may be part of an unusual pattern of activity, though this signal is unreliable and more often than not a false positive in testing, as the account had 13 transactions in the last 5 minutes in Australia, which is 2.2 times its average count per 5-minute window.",
  "recent_flags": [
    {
      "transaction_id": "7e344246-b62d-48b3-8357-4fef8d6339a6",
      "event_timestamp": "2026-08-04T02:24:09.174048",
      "amount": 44.00,
      "merchant": "Shell Gas",
      "location": "AU",
      "triggered_rules": ["velocity"],
      "injected_label": null,
      "explanation": "This transaction may be part of an unusual pattern of activity, though this signal is unreliable and more often than not a false positive in testing, as the account had 13 transactions in the last 5 minutes in Australia, which is 2.2 times its average count per 5-minute window."
    },
    {
      "transaction_id": "c4fe2ea2-1290-43ba-b0a7-6416ebc45039",
      "event_timestamp": "2026-08-04T02:23:43.020577",
      "amount": 912.06,
      "merchant": "Whole Foods",
      "location": "AU",
      "triggered_rules": ["velocity", "amount"],
      "injected_label": "amount_spike",
      "explanation": "The transaction of $912.06 at Whole Foods in Australia may be suspicious due to the account's unusually high transaction volume in the last 5 minutes, though this signal is unreliable and more often than not a false positive in testing, and it is also directly suspicious because the amount is extremely high compared to the account's historical average transaction amount."
    },
    {
      "transaction_id": "1487c985-3a85-4d52-8cde-377f9c9813b1",
      "event_timestamp": "2026-08-04T02:19:13.920274",
      "amount": 48.24,
      "merchant": "Uber",
      "location": "AU",
      "triggered_rules": ["geo"],
      "injected_label": null,
      "explanation": null
    }
    // ... 8 more, chronological. explanation is null where llm_explainer.py
    // hasn't processed that row yet — it's a batch/backfill job, not run inline.
  ]
}
```

The historical arithmetic shown in this payload must not be used for v2; current weights
are versioned and derived from conservative bounds in the checked-in release report.

Verified over the real MCP stdio protocol (tool registration → `initialize` → `call_tool`),
not just a direct function call, via a scratch MCP client.

## Known limitations

- **The Spark↔Kafka connector has intermittently stalled in the local Docker setup.** The
  progress watchdog now detects and restarts it from a durable checkpoint, but the
  underlying connector behavior is not root-caused. A managed deployment should alert on
  restart count and Kafka consumer lag.
- **`src/mcp/` shares its name with the installed `mcp` package.** Always launch it as a
  module from the repository root (`python -m src.mcp.risk_status_server`) so the local
  package does not shadow the SDK.
- **`generate_transactions.py` account identities are deterministic** (seeded by account
  index), not re-randomized per process start. An earlier version re-randomized on every
  run, which silently reassigned a given account's home country between producer
  invocations and manufactured spurious "impossible travel" between otherwise-unrelated
  runs — a real bug, not a design choice, caught while investigating an unexplained spike in
  geo false positives. Fixed; flagging in case older behavior is referenced elsewhere.
