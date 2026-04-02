.PHONY: help airflow-build airflow-init airflow-up airflow-down \
        airflow-restart airflow-logs airflow-status \
        dag-check sync-dags trigger-training \
        airflow-dev-up airflow-dev-down ml-test ml-lint data-test

PROD_COMPOSE  = docker/docker-compose.prod.yml
DEV_COMPOSE   = data-pipeline/docker/docker-compose.airflow.yml
ENV_FILE      = /opt/airflow/.env
COMPOSE_CMD   = docker compose -f $(PROD_COMPOSE) --env-file $(ENV_FILE)

help:
	@echo ""
	@echo "Boston Pulse — Root Commands"
	@echo ""
	@echo "LOCAL DEV (Mac):"
	@echo "  make airflow-dev-up      Local Airflow (data-pipeline only, fake GCS)"
	@echo "  make airflow-dev-down    Stop local Airflow"
	@echo "  make ml-test             Run ML unit tests"
	@echo "  make ml-lint             Run ML linter"
	@echo "  make data-test           Run data-pipeline tests"
	@echo ""
	@echo "PRODUCTION (GCE VM only):"
	@echo "  make airflow-build      Build image (after Dockerfile changes)"
	@echo "  make airflow-init       First-time DB setup (run once)"
	@echo "  make airflow-up         Start all containers"
	@echo "  make airflow-down       Stop all containers"
	@echo "  make airflow-restart    Restart after config changes"
	@echo "  make airflow-logs       Tail all logs"
	@echo "  make airflow-status     Show container health"
	@echo "  make dag-check          Check for DAG import errors"
	@echo "  make sync-dags          Sync DAG files to /opt/airflow/dags"
	@echo "  make trigger-training   Trigger ML training DAG now"
	@echo ""

# ── Local dev (Mac) ───────────────────────────────────────────────────────────
airflow-dev-up:
	cd data-pipeline && make airflow-up-dp

airflow-dev-down:
	cd data-pipeline && make airflow-down-dp

# ── Production (GCE VM only) ──────────────────────────────────────────────────

# Build the custom image — run after any Dockerfile or pyproject.toml change
airflow-build:
	$(COMPOSE_CMD) build --no-cache

# First time only — migrates DB and creates admin user
airflow-init:
	$(COMPOSE_CMD) up airflow-init

airflow-up:
	$(COMPOSE_CMD) up -d --remove-orphans

# Uses --env-file so no "variable not set" warnings
airflow-down:
	$(COMPOSE_CMD) down

airflow-restart:
	$(COMPOSE_CMD) up -d --remove-orphans

airflow-logs:
	$(COMPOSE_CMD) logs -f

airflow-status:
	$(COMPOSE_CMD) ps

dag-check:
	$(COMPOSE_CMD) exec airflow-scheduler \
	  airflow dags list-import-errors

sync-dags:
	@echo "Syncing DAGs from both pipelines..."
	@if [ ! -w /opt/airflow/dags ]; then \
		echo "Fixing /opt/airflow/dags permissions..."; \
		sudo chown -R $$(id -u):$$(id -g) /opt/airflow/dags; \
	fi
	rsync -av --exclude='__pycache__' --exclude='*.pyc' data-pipeline/dags/ /opt/airflow/dags/
	rsync -av --exclude='__pycache__' --exclude='*.pyc' ml/dags/ /opt/airflow/dags/
	@echo "Synced. Scheduler picks up changes in ~30s."

trigger-training:
	@echo "Training is run via GitHub Actions → Cloud Run Job (see .github/workflows/ml.yml)."
	@echo "Local one-off:  cd ml && python -m models.crime_navigate.cli train --execution-date \$$(date +%Y-%m-%d) --stage staging"

# ── Tests ─────────────────────────────────────────────────────────────────────
ml-test:
	cd ml && make test

ml-lint:
	cd ml && make lint

data-test:
	cd data-pipeline && make test