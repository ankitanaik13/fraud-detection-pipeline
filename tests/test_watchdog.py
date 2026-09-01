from src.streaming.watchdog import ProgressWatchdog


def test_watchdog_resets_on_new_progress():
    watchdog = ProgressWatchdog(timeout_seconds=30)
    assert watchdog.observe(None, 0) is False
    assert watchdog.observe("batch-1", 10) is False
    assert watchdog.observe("batch-1", 35) is False
    assert watchdog.observe("batch-1", 41) is True
    assert watchdog.observe("batch-2", 42) is False


def test_watchdog_detects_no_initial_progress():
    watchdog = ProgressWatchdog(timeout_seconds=10)
    assert watchdog.observe(None, 5) is False
    assert watchdog.observe(None, 16) is True
