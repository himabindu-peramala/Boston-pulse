# Boston Pulse — Root Makefile
# Thin orchestration layer. Each component has its own Makefile.
#
# LOCAL DEV:   run on your Mac
# PRODUCTION:  run on GCE VM only

.PHONY: help \
        airflow-dev-up airflow-dev-down \
        airflow-init airflow-up airflow-down \
        airflow-restart airflow-logs airflow-status \
        dag-check sync-dags trigger-training \
        ml-test ml-lint data-test

PROD_COMPOSE = docker/docker-compose.prod.yml
ENV_FILE     = /opt/airflow/.env

help:
	@echo ""
	@echo "Boston Pulse — Root Commands"
	@echo ""
	@echo "LOCAL DEV (your Mac):"
	@echo "  make airflow-dev-up      Local Airflow (data-pipeline only, fake GCS)"
	@echo "  make airflow-dev-down    Stop local Airflow"
	@echo "  make ml-test             Run ML unit tests"
	@echo "  make ml-lint             Run ML linter"
	@echo "  make data-test           Run data-pipeline tests"
	@echo ""
	@echo "PRODUCTION (GCE VM only):"
	@echo "  make airflow-init        First-time DB setup and user creation"
	@echo "  make airflow-up          Start Airflow (both pipelines)"
	@echo "  make airflow-down        Stop Airflow"
	@echo "  make airflow-restart     Restart after compose/env changes"
	@echo "  make airflow-logs        Tail all container logs"
	@echo "  make airflow-status      Show container health"
	@echo "  make dag-check           Check for DAG import errors"
	@echo "  make sync-dags           Manually sync DAG files"
	@echo "  make trigger-training    Trigger ML training DAG now"
	@echo ""

# ── Local dev ─────────────────────────────────────────────────────────────────
airflow-dev-up:
	cd data-pipeline && make airflow-up-dp

airflow-dev-down:
	cd data-pipeline && make airflow-down-dp

# ── Production (VM only) ──────────────────────────────────────────────────────
airflow-init:
	docker compose -f $(PROD_COMPOSE) --env-file $(ENV_FILE) up airflow-init

airflow-up:
	docker compose -f $(PROD_COMPOSE) --env-file $(ENV_FILE) up -d --remove-orphans

airflow-down:
	docker compose -f $(PROD_COMPOSE) down

airflow-restart:
	docker compose -f $(PROD_COMPOSE) --env-file $(ENV_FILE) up -d --remove-orphans

airflow-logs:
	docker compose -f $(PROD_COMPOSE) logs -f

airflow-status:
	docker compose -f $(PROD_COMPOSE) ps

dag-check:
	docker compose -f $(PROD_COMPOSE) \
	  exec airflow-scheduler \
	  airflow dags list-import-errors

sync-dags:
	@echo "Syncing DAGs from both pipelines..."
	cp -r data-pipeline/dags/* /opt/airflow/dags/ 2>/dev/null || true
	cp -r ml/dags/* /opt/airflow/dags/ 2>/dev/null || true
	@echo "Done. Scheduler picks up changes in ~30s."

trigger-training:
	docker compose -f $(PROD_COMPOSE) \
	  exec airflow-scheduler \
	  airflow dags trigger crime_navigate_train \
	  --conf "{\"execution_date\": \"$$(date +%Y-%m-%d)\"}"

# ── Component tests ───────────────────────────────────────────────────────────
ml-test:
	cd ml && make test

ml-lint:
	cd ml && make lint

data-test:
	cd data-pipeline && make test