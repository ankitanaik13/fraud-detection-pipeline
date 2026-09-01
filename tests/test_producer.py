from datetime import UTC, datetime

from src.producer.generate_transactions import build_accounts, touch_account


def test_run_namespace_isolates_account_state_between_evaluations():
    first = build_accounts(1, namespace="first")[0]
    second = build_accounts(1, namespace="second")[0]

    assert first.account_id != second.account_id
    assert first.home_country == second.home_country
    assert first.historical_avg == second.historical_avg


def test_impossible_travel_does_not_create_unlabeled_return_home():
    account = build_accounts(1)[0]
    event = {
        "location": "GB" if account.home_country != "GB" else "US",
        "timestamp": datetime.now(UTC).isoformat(),
        "injected_label": "impossible_travel",
    }

    touch_account(account, event)

    assert account.home_country == event["location"]
    assert account.last_location == event["location"]
