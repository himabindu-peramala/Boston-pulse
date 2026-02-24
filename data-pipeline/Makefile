# Common commands for development workflow.
# Run `make help` to see all available commands.
# =============================================================================

.PHONY: help setup setup-dev install lint format test test-unit test-integration \
        test-schema test-cov clean airflow-up airflow-down airflow-logs \
        airflow-up-dp airflow-down-dp airflow-logs-dp \
        create-dataset docker-build docker-push validate-dags

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
BLACK := $(PYTHON) -m black
MYPY := $(PYTHON) -m mypy

# Directories
SRC_DIR := src
TESTS_DIR := tests
DAGS_DIR := dags
SCHEMAS_DIR := schemas
CONFIGS_DIR := configs

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# =============================================================================
# HELP
# =============================================================================

help:  ## Show this help message
	@echo ""
	@echo "$(BLUE)Boston Pulse Data Pipeline$(NC)"
	@echo "$(BLUE)===========================$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'setup|install' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'lint|format|test|clean' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Airflow Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'airflow|dag' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Docker Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'docker' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Dataset Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'dataset' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# SETUP
# =============================================================================

setup:  ## One-command setup for local development
	@echo "$(BLUE)Setting up Boston Pulse Data Pipeline...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Setup complete! Run 'make test' to verify.$(NC)"

setup-dev:  ## Setup with all development dependencies
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,streaming]"
	pre-commit install
	@echo "$(GREEN)Development setup complete!$(NC)"

install:  ## Install production dependencies only
	$(PIP) install --upgrade pip
	$(PIP) install -e .

# =============================================================================
# LINTING & FORMATTING
# =============================================================================

lint:  ## Run all linters (ruff, black check, mypy)
	@echo "$(BLUE)Running Ruff...$(NC)"
	$(RUFF) check $(SRC_DIR) $(DAGS_DIR) $(TESTS_DIR)
	@echo "$(BLUE)Checking Black formatting...$(NC)"
	$(BLACK) --check --diff $(SRC_DIR) $(DAGS_DIR) $(TESTS_DIR)
	@echo "$(GREEN)All lint checks passed!$(NC)"

lint-fix:  ## Run linters and auto-fix issues
	@echo "$(BLUE)Fixing with Ruff...$(NC)"
	$(RUFF) check --fix $(SRC_DIR) $(DAGS_DIR) $(TESTS_DIR)
	@echo "$(BLUE)Formatting with Black...$(NC)"
	$(BLACK) $(SRC_DIR) $(DAGS_DIR) $(TESTS_DIR)
	@echo "$(GREEN)Lint fixes applied!$(NC)"

format:  ## Format code with Black and Ruff
	$(RUFF) check --fix $(SRC_DIR) $(DAGS_DIR) $(TESTS_DIR)
	$(BLACK) $(SRC_DIR) $(DAGS_DIR) $(TESTS_DIR)

typecheck:  ## Run MyPy type checking
	@echo "$(BLUE)Running MyPy...$(NC)"
	$(MYPY) $(SRC_DIR)

# =============================================================================
# TESTING
# =============================================================================

test:  ## Run all tests
	$(PYTEST) $(TESTS_DIR) -v

test-unit:  ## Run unit tests only
	$(PYTEST) $(TESTS_DIR)/unit -v -m "unit or not integration"

test-integration:  ## Run integration tests only
	$(PYTEST) $(TESTS_DIR)/integration -v -m integration

test-schema:  ## Run schema validation tests
	$(PYTEST) $(TESTS_DIR)/schema_tests -v -m schema

test-bias:  ## Run bias detection tests
	$(PYTEST) $(TESTS_DIR)/bias_tests -v

test-cov:  ## Run tests with coverage report
	$(PYTEST) $(TESTS_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=80
	@echo "$(GREEN)Coverage report generated at htmlcov/index.html$(NC)"

test-fast:  ## Run tests excluding slow ones
	$(PYTEST) $(TESTS_DIR) -v -m "not slow"

# =============================================================================
# AIRFLOW
# =============================================================================

airflow-up:  ## Start local Airflow environment
	@echo "$(BLUE)Starting Airflow...$(NC)"
	cd ../docker && docker-compose -f docker-compose.airflow.yml up -d
	@echo "$(GREEN)Airflow is starting at http://localhost:8080$(NC)"
	@echo "$(YELLOW)Default credentials: airflow/airflow$(NC)"

airflow-down:  ## Stop local Airflow environment
	@echo "$(BLUE)Stopping Airflow...$(NC)"
	cd ../docker && docker-compose -f docker-compose.airflow.yml down
	@echo "$(GREEN)Airflow stopped.$(NC)"

airflow-logs:  ## View Airflow logs
	cd ../docker && docker-compose -f docker-compose.airflow.yml logs -f

airflow-up-dp:  ## Start Airflow (compose in data-pipeline/docker)
	@echo "$(BLUE)Starting Airflow (data-pipeline/docker)...$(NC)"
	cd docker && docker compose --env-file ../.env -f docker-compose.airflow.yml up -d --build
	@echo "$(GREEN)Airflow is starting at http://localhost:8080$(NC)"

airflow-down-dp:  ## Stop Airflow (compose in data-pipeline/docker)
	@echo "$(BLUE)Stopping Airflow (data-pipeline/docker)...$(NC)"
	cd docker && docker compose --env-file ../.env -f docker-compose.airflow.yml down
	@echo "$(GREEN)Airflow stopped.$(NC)"

airflow-logs-dp:  ## View Airflow logs (compose in data-pipeline/docker)
	cd docker && docker compose --env-file ../.env -f docker-compose.airflow.yml logs -f

airflow-restart:  ## Restart Airflow
	$(MAKE) airflow-down
	$(MAKE) airflow-up

validate-dags:  ## Validate all Airflow DAGs
	@echo "$(BLUE)Validating DAGs...$(NC)"
	$(PYTHON) -c "\
import sys; \
from pathlib import Path; \
sys.path.insert(0, str(Path.cwd())); \
from airflow.models import DagBag; \
dag_bag = DagBag(dag_folder='$(DAGS_DIR)', include_examples=False); \
errors = dag_bag.import_errors; \
if errors: \
    print('DAG Import Errors:'); \
    [print(f'  {k}: {v}') for k,v in errors.items()]; \
    sys.exit(1); \
print(f'Successfully validated {len(dag_bag.dags)} DAGs'); \
[print(f'  - {d}') for d in sorted(dag_bag.dags.keys())]"

# =============================================================================
# DOCKER
# =============================================================================

docker-build:  ## Build the data pipeline Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t boston-pulse-pipeline:latest -f ../docker/Dockerfile.pipeline .

docker-push:  ## Push Docker image to registry (requires GCR_PROJECT env var)
	@if [ -z "$(GCR_PROJECT)" ]; then \
		echo "$(RED)Error: GCR_PROJECT environment variable not set$(NC)"; \
		exit 1; \
	fi
	docker tag boston-pulse-pipeline:latest gcr.io/$(GCR_PROJECT)/boston-pulse-pipeline:latest
	docker push gcr.io/$(GCR_PROJECT)/boston-pulse-pipeline:latest

# =============================================================================
# DATASET SCAFFOLDING
# =============================================================================

create-dataset:  ## Create a new dataset scaffold (usage: make create-dataset NAME=foo)
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME is required. Usage: make create-dataset NAME=my_dataset$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Creating dataset scaffold for '$(NAME)'...$(NC)"
	@bash ../scripts/create_dataset.sh $(NAME)
	@echo "$(GREEN)Dataset '$(NAME)' scaffold created!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Implement $(SRC_DIR)/datasets/$(NAME)/ingest.py"
	@echo "  2. Implement $(SRC_DIR)/datasets/$(NAME)/preprocess.py"
	@echo "  3. Implement $(SRC_DIR)/datasets/$(NAME)/features.py"
	@echo "  4. Update $(SCHEMAS_DIR)/$(NAME)/*.json schemas"
	@echo "  5. Configure $(CONFIGS_DIR)/datasets/$(NAME).yaml"
	@echo "  6. Run 'make test-unit' to verify"

# =============================================================================
# VALIDATION
# =============================================================================

validate-schemas:  ## Validate all JSON schemas
	@echo "$(BLUE)Validating JSON schemas...$(NC)"
	$(PYTHON) -c "\
import json; \
from pathlib import Path; \
errors = []; \
for f in Path('$(SCHEMAS_DIR)').rglob('*.json'): \
    try: json.load(open(f)); print(f'  ✓ {f}'); \
    except Exception as e: errors.append(f'{f}: {e}'); \
if errors: [print(f'  ✗ {e}') for e in errors]; exit(1); \
print('All schemas valid!')"

validate-configs:  ## Validate all YAML configs
	@echo "$(BLUE)Validating YAML configs...$(NC)"
	$(PYTHON) -c "\
import yaml; \
from pathlib import Path; \
errors = []; \
for f in Path('$(CONFIGS_DIR)').rglob('*.yaml'): \
    try: yaml.safe_load(open(f)); print(f'  ✓ {f}'); \
    except Exception as e: errors.append(f'{f}: {e}'); \
if errors: [print(f'  ✗ {e}') for e in errors]; exit(1); \
print('All configs valid!')"

# =============================================================================
# CLEANING
# =============================================================================

clean:  ## Remove build artifacts and caches
	@echo "$(BLUE)Cleaning...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean  ## Remove all artifacts including Airflow home
	rm -rf airflow_home/
