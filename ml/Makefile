# Boston Pulse ML Training Pipeline - Makefile

.PHONY: install install-dev test lint format typecheck train clean mlflow-local help

# Default target
help:
	@echo "Boston Pulse ML Training Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install with dev dependencies"
	@echo "  make test          Run unit tests with coverage"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code with black"
	@echo "  make typecheck     Run mypy type checking"
	@echo "  make train         Run local training (requires GCS access)"
	@echo "  make mlflow-local  Start local MLflow server"
	@echo "  make clean         Remove build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/unit/ -v \
		--cov=models \
		--cov=shared \
		--cov-report=term-missing \
		--cov-fail-under=80

test-integration:
	pytest tests/integration/ -v --timeout=300

test-all:
	pytest tests/ -v --cov=models --cov=shared --cov-report=html

# Code quality
lint:
	ruff check models/ shared/ dags/ tests/

lint-fix:
	ruff check models/ shared/ dags/ tests/ --fix

format:
	black models/ shared/ dags/ tests/

format-check:
	black --check models/ shared/ dags/ tests/

typecheck:
	mypy models/ shared/ --ignore-missing-imports

# Local development
mlflow-local:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow-artifacts \
		--host 0.0.0.0 --port 5001

train:
	@echo "Running local training..."
	python -c "from dags.crime_navigate_train_dag import task_load_features; print('DAG imports OK')"

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage
	rm -rf mlflow.db mlflow-artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
