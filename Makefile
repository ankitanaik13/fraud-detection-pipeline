.PHONY: install lint test check up down evaluate

install:
	python -m pip install -r requirements-dev.txt

lint:
	python -m ruff check .

test:
	python -m pytest -q \
		--cov=src.common --cov=src.detection --cov=src.evaluation.metrics \
		--cov=src.streaming.watchdog --cov=src.storage.db_writer \
		--cov-report=term-missing --cov-fail-under=85

check: lint test

up:
	docker compose up -d --build

down:
	docker compose down

evaluate:
	test -n "$(RUN_ID)" || (echo "Usage: make evaluate RUN_ID=<evaluation-run-id>" && exit 2)
	python -m src.evaluation.evaluate_database --run-id "$(RUN_ID)"
