.PHONY: lockcheck format lint typecheck unit test smoke capacity check

lockcheck:
	uv lock --check

format:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy recommend

unit:
	uv run pytest recommend/train/tests/unit -q

test:
	uv run pytest recommend/train/tests -q -m "not performance"

smoke:
	uv run pytest recommend/train/tests/smoke -q

capacity:
	uv run pytest recommend/train/tests/performance -q -m performance

check: lockcheck format lint typecheck test tf-check

.PHONY: tf-check
tf-check:
	env -u UV_PROJECT_ENVIRONMENT uv lock --project recommend_tf --check
	env -u UV_PROJECT_ENVIRONMENT PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 uv run --project recommend_tf --locked python -m pytest recommend_tf/train/tests -q -p no:cacheprovider --disable-warnings
