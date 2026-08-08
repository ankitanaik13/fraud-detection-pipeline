# Real-Time Transaction Fraud Detection Pipeline

A local, four-stage pipeline that simulates a live transaction stream, flags anomalies with
per-account stateful rules in Spark, explains each flag in plain English via an LLM, and
exposes account risk status as an MCP tool. Runs entirely on Docker Compose — no GPU, no
notebooks, no cloud dependency except the LLM call itself.

Built as a portfolio project to demonstrate the full loop: synthetic ground truth →
streaming detection → measured precision/recall → tuning against real numbers → LLM
enrichment → a queryable tool interface. The results below are real, including the ones
that aren't flattering (velocity's recall is bad, and here's the swept threshold data
proving why).

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
docker compose up -d kafka postgres spark
python src/producer/generate_transactions.py --rate 8

# in another shell
docker compose exec spark spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9,org.postgresql:postgresql:42.7.3 \
  /opt/spark-apps/streaming/fraud_detector.py

# once some transactions are flagged
python src/explain/llm_explainer.py --limit 50
python src/mcp/risk_status_server.py   # stdio transport
```

`generate_transactions.py --injection-mode {random,borderline,straddle}` controls how
`impossible_travel` anomalies are constructed — see "The geo finding" below for why this
matters and what each mode is for.

## Detection rules (`src/streaming/fraud_detector.py`)

Per-account rolling state, maintained via `applyInPandasWithState`:

| Rule | Trips when | Needs |
|---|---|---|
| **amount** | amount > 5x the account's running historical average | ≥3 prior transactions |
| **geo** | implied travel speed between consecutive transaction locations > 900 km/h | a previous transaction to compare against |
| **velocity** | live 5-minute transaction count > 1.6x the account's average count across its own past *completed* 5-minute windows | ≥1 completed window |

Velocity's baseline deliberately excludes the in-progress window — folding a burst's own
events into its own baseline as they happen would let the anomaly dilute the very average
it's being compared against. (This was a real bug caught mid-development; see below.)

## Results (20-minute soak run, 500 simulated accounts, 10,028 events)

| Rule | Precision | Recall |
|---|---|---|
| amount | 100% | 77% |
| geo | 50% | 94% |
| velocity | 26% | 28% |
| overall (any rule) | 38% | 43% |

### The velocity finding

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

No threshold gets both good precision and good recall — precision tops out around 26-28%
no matter where the line is drawn. **Root cause:** a 4-8 event injected burst only pushes an
account's 5-minute window count to ~1.5x its own baseline on average (max observed 5x);
normal (non-burst) windows already sit at ~1.0-1.1x baseline just from ordinary variance.
The signal and the noise floor are too close together for a ratio-to-mean test to separate
them cleanly. Two ways to actually fix it: inject larger bursts (more separation from
baseline), or replace the fixed-multiplier test with a variance-aware statistical test
(z-score against the window-count distribution, not just its mean).

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
  validates the producer's back-calculation and the detector's haversine math (duplicated
  across two files) agree with each other precisely.

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
python src/explain/llm_explainer.py --limit 50
python src/explain/llm_explainer.py --transaction-ids id1,id2   # re-explain specific rows
```

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

- `risk_score` — sum of each triggered rule's precision across all of the account's flags
  (an amount flag contributes ~1.00, a geo flag ~0.50, a velocity flag ~0.26), so the score
  reflects trustworthy signal, not just flag count — an account with nine velocity flags and
  one amount flag scores lower than one with two amount flags
- `risk_level` — `none` / `low` / `medium` / `high`, thresholded on `risk_score`
- `flags_by_rule`, `total_flags`, `total_processed_transactions`
- `recent_flags` — up to the last 20 flagged transactions, each with its triggered rule(s),
  injected label (ground truth, for dev/testing), and LLM explanation where generated

```bash
python src/mcp/risk_status_server.py   # stdio transport
```

### Sample output

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

Notice the risk score arithmetic: 9 velocity flags contribute only ~2.34 (9 × 0.26) while
the single amount flag contributes 1.00 — volume of low-precision flags doesn't dominate the
score the way a naive count would.

Verified over the real MCP stdio protocol (tool registration → `initialize` → `call_tool`),
not just a direct function call, via a scratch MCP client.

## Known limitations

- **The Spark↔Kafka connector intermittently stalls mid-stream** in this local setup — the
  stream-execution thread keeps looping but stops issuing new micro-batches, with zero new
  Spark jobs launched. Applying `spark.sql.streaming.kafka.useDeprecatedOffsetFetching=true`
  helped but didn't eliminate it. **Workaround used throughout development:** batch-replay
  the full topic (`startingOffsets=earliest`) rather than babysitting a long-running live
  query — this consistently completes in seconds without stalling, at the cost of not being
  a truly continuous live deployment. Not yet root-caused; a real continuous deployment
  would need this fixed or a supervisor that restarts the job on stall.
- **`src/mcp/` shares its name with the installed `mcp` package.** Running
  `risk_status_server.py` directly (as shown above) is unaffected, but avoid adding `src/`
  itself to `sys.path` in code that also needs to `import mcp` (the third-party SDK) — the
  local package would shadow it.
- **`generate_transactions.py` account identities are deterministic** (seeded by account
  index), not re-randomized per process start. An earlier version re-randomized on every
  run, which silently reassigned a given account's home country between producer
  invocations and manufactured spurious "impossible travel" between otherwise-unrelated
  runs — a real bug, not a design choice, caught while investigating an unexplained spike in
  geo false positives. Fixed; flagging in case older behavior is referenced elsewhere.
