# Shortcuts so nobody has to remember long commands.
# Usage: `make test`, `make lint`, `make ci` (runs what GitHub runs).
# On Windows, run these inside WSL or Git Bash, or just copy the command after the tab.

.PHONY: install test integration lint format ci clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

integration:
	pytest -m integration

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

# Same checks the GitHub pipeline runs. If this passes locally, CI will almost always pass.
ci: lint test

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info htmlcov .coverage
