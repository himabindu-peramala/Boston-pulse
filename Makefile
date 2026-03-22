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
	@echo "LOCAL DEV (Mac):    make airflow-dev-up / airflow-dev-down"
	@echo "PRODUCTION (VM):"
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

# ── Local dev ─────────────────────────────────────────────────────────────────
airflow-dev-up:
	cd data-pipeline && make airflow-up-dp

airflow-dev-down:
	cd data-pipeline && make airflow-down-dp

# ── Production ────────────────────────────────────────────────────────────────

# Build the custom image — run after any Dockerfile change
airflow-build:
	$(COMPOSE_CMD) build --no-cache

# First time only — sets up DB and admin user
airflow-init:
	$(COMPOSE_CMD) up airflow-init

airflow-up:
	$(COMPOSE_CMD) up -d --remove-orphans

airflow-down:
	docker compose -f $(PROD_COMPOSE) down

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
	@echo "Syncing DAGs..."
	cp -r data-pipeline/dags/* /opt/airflow/dags/ 2>/dev/null || true
	cp -r ml/dags/* /opt/airflow/dags/ 2>/dev/null || true
	@echo "Synced. Scheduler picks up changes in ~30s."

trigger-training:
	$(COMPOSE_CMD) exec airflow-scheduler \
	  airflow dags trigger crime_navigate_train \
	  --conf "{\"execution_date\": \"$$(date +%Y-%m-%d)\"}"

# ── Tests ─────────────────────────────────────────────────────────────────────
ml-test:
	cd ml && make test

ml-lint:
	cd ml && make lint

data-test:
	cd data-pipeline && make test