.PHONY: format lint typecheck unit test smoke capacity check

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

check: format lint typecheck test
