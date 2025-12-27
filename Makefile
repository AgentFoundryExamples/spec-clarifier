.PHONY: help install install-dev format lint type-check test test-verbose clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install runtime dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Type check with mypy"
	@echo "  make test          - Run tests with pytest"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make clean         - Remove cache and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	black app/ tests/

lint:
	ruff check app/ tests/

type-check:
	mypy app/

test:
	pytest

test-verbose:
	pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
