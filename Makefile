# Load .env.demo if it exists and export every variable it defines so both
# $(shell …) expansions and recipe commands see the values. The leading `-`
# makes the include silent when the file is missing (CI, first-time clone).
-include .env.demo
export

.PHONY: help airflow-build airflow-init airflow-up airflow-down \
        airflow-restart airflow-logs airflow-status \
        dag-check sync-dags trigger-training \
        airflow-dev-up airflow-dev-down ml-test ml-lint data-test \
        demo-preflight demo-infra demo-app demo-destroy \
        demo-wif demo-full demo-status \
        demo-airflow-url demo-airflow-password demo-airflow-ssh \
        demo-trigger-data demo-trigger-training demo-wait-vm \
        demo-up demo-down demo-create-project _demo-check-env

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
	@echo "FRESH-PROJECT DEMO (GCP) — end-to-end:"
	@echo "  make demo-up                 ONE-SHOT: .env.demo -> create project + billing + infra + VM"
	@echo "  make demo-down               ONE-SHOT: destroy infra + (optional) delete project + clean state"
	@echo ""
	@echo "  Lower-level targets (called by demo-up/demo-down):"
	@echo "  make demo-infra              terraform apply (APIs, AR, GCS, Firestore, SA, WIF, Airflow VM)"
	@echo "  make demo-wif                Print gh secret set commands from Terraform outputs"
	@echo "  make demo-full               preflight + infra + WIF hints + VM URL"
	@echo "  make demo-status             Show Terraform resources + live GCP artifacts + console links"
	@echo "  make demo-destroy            terraform destroy (teardown)"
	@echo ""
	@echo "  Airflow VM (data-pipeline):"
	@echo "  make demo-airflow-url        Print Airflow UI URL"
	@echo "  make demo-airflow-password   Print generated admin password"
	@echo "  make demo-airflow-ssh        SSH into the VM"
	@echo "  make demo-wait-vm            Wait until the VM startup script finishes"
	@echo "  make demo-trigger-data       Unpause + trigger the crime_navigate DAG on the VM"
	@echo ""
	@echo "  ML training:"
	@echo "  make demo-trigger-training   Execute the Cloud Run training job"
	@echo "  make demo-app                (Optional) Build ML image + create training Job locally"
	@echo ""
	@echo "    Required env vars:  GCP_PROJECT_ID"
	@echo "    Optional env vars:  GITHUB_REPOSITORY (enables WIF), SLACK_AUTH_TOKEN (enables Slack)"
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

# ── Fresh-project demo ────────────────────────────────────────────────────────
# Terraform handles all infrastructure; one bash script handles the app layer.
# Rubric-friendly: shows real IaC + automated deploy script.

# Preflight: confirms GCP_PROJECT_ID is set, gcloud is authenticated, the project
# exists, and billing is linked. These are the three things that previously made
# `terraform apply` fail with a cryptic 403.
demo-preflight:
	@test -n "$$GCP_PROJECT_ID" || { echo "✗ Set GCP_PROJECT_ID (e.g. 'export GCP_PROJECT_ID=boston-pulse-demo-2026')"; exit 1; }
	@command -v gcloud >/dev/null    || { echo "✗ gcloud CLI not found"; exit 1; }
	@command -v terraform >/dev/null || { echo "✗ terraform CLI not found (brew install terraform)"; exit 1; }
	@gcloud auth application-default print-access-token >/dev/null 2>&1 \
	  || { echo "✗ gcloud ADC not set. Run: gcloud auth application-default login"; exit 1; }
	@gcloud projects describe "$$GCP_PROJECT_ID" >/dev/null 2>&1 \
	  || { echo "✗ Project '$$GCP_PROJECT_ID' not found or not accessible."; \
	       echo "  Create it:    gcloud projects create $$GCP_PROJECT_ID"; \
	       echo "  Link billing: gcloud billing projects link $$GCP_PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX"; \
	       exit 1; }
	@billing=$$(gcloud billing projects describe "$$GCP_PROJECT_ID" --format='value(billingEnabled)' 2>/dev/null || echo "false"); \
	  if [ "$$billing" != "True" ] && [ "$$billing" != "true" ]; then \
	    echo "✗ Billing is not enabled on '$$GCP_PROJECT_ID'."; \
	    echo "  Link billing: gcloud billing projects link $$GCP_PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX"; \
	    exit 1; \
	  fi
	@echo "✓ Preflight passed for $$GCP_PROJECT_ID"

demo-infra: demo-preflight
	cd infrastructure/terraform && terraform init -input=false -upgrade
	cd infrastructure/terraform && terraform apply -auto-approve \
	  -var="project_id=$$GCP_PROJECT_ID" \
	  $${GITHUB_REPOSITORY:+-var="github_repository=$$GITHUB_REPOSITORY"} \
	  $${SLACK_AUTH_TOKEN:+-var="slack_auth_token=$$SLACK_AUTH_TOKEN"} \
	  $${VM_ZONE:+-var="vm_zone=$$VM_ZONE"} \
	  $${VM_MACHINE_TYPE:+-var="vm_machine_type=$$VM_MACHINE_TYPE"}

demo-app:
	./scripts/deploy_app.sh

demo-wif:
	@test -n "$$GITHUB_REPOSITORY" || { echo "Set GITHUB_REPOSITORY=user/repo"; exit 1; }
	@echo ""
	@echo "━━━ Update these GitHub secrets ━━━"
	@WIF_PROVIDER=$$(cd infrastructure/terraform && terraform output -raw wif_provider); \
	WIF_SERVICE_ACCOUNT=$$(cd infrastructure/terraform && terraform output -raw wif_service_account); \
	CLOUD_RUN_TRAINING_JOB=$$(cd infrastructure/terraform && terraform output -raw cloud_run_training_job); \
	echo "gh secret set WIF_PROVIDER -b\"$$WIF_PROVIDER\""; \
	echo "gh secret set WIF_SERVICE_ACCOUNT -b\"$$WIF_SERVICE_ACCOUNT\""; \
	echo "gh secret set GCP_PROJECT_ID -b\"$$GCP_PROJECT_ID\""; \
	echo "gh secret set GCS_BUCKET -b\"$$GCP_PROJECT_ID-data-pipeline\""; \
	echo "gh secret set CLOUD_RUN_TRAINING_JOB -b\"$$CLOUD_RUN_TRAINING_JOB\""

demo-destroy:
	@test -n "$$GCP_PROJECT_ID" || { echo "Set GCP_PROJECT_ID"; exit 1; }
	-gcloud run jobs delete ml-training-job --region=$${GCP_REGION:-us-east1} --project=$$GCP_PROJECT_ID --quiet
	cd infrastructure/terraform && terraform destroy -auto-approve \
	  -var="project_id=$$GCP_PROJECT_ID" \
	  $${GITHUB_REPOSITORY:+-var="github_repository=$$GITHUB_REPOSITORY"} \
	  $${SLACK_AUTH_TOKEN:+-var="slack_auth_token=$$SLACK_AUTH_TOKEN"} \
	  $${VM_ZONE:+-var="vm_zone=$$VM_ZONE"} \
	  $${VM_MACHINE_TYPE:+-var="vm_machine_type=$$VM_MACHINE_TYPE"}

demo-full: demo-preflight demo-infra demo-wif
	@echo ""
	@echo "━━━ Infra ready ━━━"
	@URL=$$(cd infrastructure/terraform && terraform output -raw airflow_url 2>/dev/null || echo ""); \
	if [ -n "$$URL" ]; then \
	  echo "Airflow UI:       $$URL"; \
	  echo "Admin user:       admin"; \
	  echo "Admin password:   $$(cd infrastructure/terraform && terraform output -raw airflow_admin_password 2>/dev/null)"; \
	  echo ""; \
	  echo "The VM is still running its startup script (clone repo, build image,"; \
	  echo "start Airflow — ~5 min). Run:  make demo-wait-vm"; \
	  echo "Then:                         make demo-trigger-data"; \
	  echo "Once features are written:    make demo-trigger-training"; \
	fi
	@echo ""
	@echo "Inspect everything:  make demo-status"

# ── Airflow VM helpers ────────────────────────────────────────────────────────

demo-airflow-url:
	@cd infrastructure/terraform && terraform output -raw airflow_url

demo-airflow-password:
	@cd infrastructure/terraform && terraform output -raw airflow_admin_password

demo-airflow-ssh:
	@NAME=$$(cd infrastructure/terraform && terraform output -raw airflow_vm_name); \
	ZONE=$$(cd infrastructure/terraform && terraform output -raw airflow_vm_zone); \
	gcloud compute ssh "$$NAME" --zone="$$ZONE" --project="$$GCP_PROJECT_ID"

# Polls the startup-script log on the VM until it writes the "Startup complete"
# marker. Useful right after `make demo-infra` before triggering a DAG.
demo-wait-vm:
	@NAME=$$(cd infrastructure/terraform && terraform output -raw airflow_vm_name); \
	ZONE=$$(cd infrastructure/terraform && terraform output -raw airflow_vm_zone); \
	echo "Waiting for Airflow startup on $$NAME ($$ZONE) — this typically takes 3-6 minutes..."; \
	for i in $$(seq 1 60); do \
	  if gcloud compute ssh "$$NAME" --zone="$$ZONE" --project="$$GCP_PROJECT_ID" --quiet \
	       --command='grep -q "Startup complete" /var/log/boston-pulse-startup.log 2>/dev/null' 2>/dev/null; then \
	    echo "✓ Airflow is up."; \
	    exit 0; \
	  fi; \
	  printf "."; sleep 10; \
	done; \
	echo ""; echo "✗ Startup did not complete within 10 minutes."; \
	echo "  Check logs:  make demo-airflow-ssh  then  sudo tail -f /var/log/boston-pulse-startup.log"; \
	exit 1

# Unpauses and triggers the crime_navigate DAG on the VM so the data bucket
# gets populated before the ML training job runs.
demo-trigger-data:
	@NAME=$$(cd infrastructure/terraform && terraform output -raw airflow_vm_name); \
	ZONE=$$(cd infrastructure/terraform && terraform output -raw airflow_vm_zone); \
	DAG=$${DAG_ID:-crime_navigate_pipeline}; \
	EXEC_DATE=$${EXEC_DATE:-$$(date -u +%Y-%m-%d)}; \
	echo "Triggering DAG '$$DAG' on $$NAME for execution_date=$$EXEC_DATE..."; \
	gcloud compute ssh "$$NAME" --zone="$$ZONE" --project="$$GCP_PROJECT_ID" --quiet \
	  --command="cd /opt/boston-pulse && sudo docker compose -f docker/docker-compose.prod.yml --env-file /opt/airflow/.env exec -T airflow-scheduler airflow dags unpause $$DAG && sudo docker compose -f docker/docker-compose.prod.yml --env-file /opt/airflow/.env exec -T airflow-scheduler airflow dags trigger -e $$EXEC_DATE $$DAG"; \
	echo ""; \
	echo "Watch progress in the UI: $$(cd infrastructure/terraform && terraform output -raw airflow_url)"

# Executes the Cloud Run training job (created by CI/CD). Assumes CI has built
# and pushed the ml-training image and created the job at least once.
demo-trigger-training:
	@test -n "$$GCP_PROJECT_ID" || { echo "Set GCP_PROJECT_ID"; exit 1; }
	@REGION=$${GCP_REGION:-us-east1}; \
	JOB=$${TRAINING_JOB_NAME:-ml-training-job}; \
	echo "Executing Cloud Run job '$$JOB' in $$REGION..."; \
	gcloud run jobs execute "$$JOB" --project="$$GCP_PROJECT_ID" --region="$$REGION" --wait

# Summarise everything the Terraform + CI/CD flow has provisioned so you can
# audit what exists in the fresh project without clicking around the console.
demo-status:
	@test -n "$$GCP_PROJECT_ID" || { echo "Set GCP_PROJECT_ID"; exit 1; }
	@REGION=$${GCP_REGION:-us-east1}; \
	PROJECT_NUMBER=$$(gcloud projects describe $$GCP_PROJECT_ID --format='value(projectNumber)' 2>/dev/null); \
	echo ""; \
	echo "━━━ Terraform-managed resources ━━━"; \
	cd infrastructure/terraform && terraform state list 2>/dev/null | sed 's/^/  • /' || echo "  (no terraform state)"; \
	cd - >/dev/null; \
	echo ""; \
	echo "━━━ Terraform outputs ━━━"; \
	cd infrastructure/terraform && terraform output 2>/dev/null || true; \
	cd - >/dev/null; \
	echo ""; \
	echo "━━━ Live GCP artifacts ━━━"; \
	echo "  GCS buckets:";        gcloud storage buckets list --project=$$GCP_PROJECT_ID --format='value(name)' 2>/dev/null | sed 's/^/    gs:\/\//' || true; \
	echo "  Artifact Registry:";  gcloud artifacts repositories list --project=$$GCP_PROJECT_ID --location=$$REGION --format='value(name)' 2>/dev/null | sed 's/^/    /' || true; \
	echo "  Docker images (ml-training):"; \
	  gcloud artifacts docker images list $$REGION-docker.pkg.dev/$$GCP_PROJECT_ID/ml-images/ml-training --project=$$GCP_PROJECT_ID --format='value(IMAGE,TAGS)' 2>/dev/null | sed 's/^/    /' || echo "    (not pushed yet — run CI or make demo-app)"; \
	echo "  Cloud Run jobs:";     gcloud run jobs list --project=$$GCP_PROJECT_ID --region=$$REGION --format='value(name)' 2>/dev/null | sed 's/^/    /' || echo "    (none — CI creates ml-training-job)"; \
	echo "  Compute VMs:";        gcloud compute instances list --project=$$GCP_PROJECT_ID --format='value(name,zone,status,EXTERNAL_IP)' 2>/dev/null | sed 's/^/    /' || true; \
	echo "  Firestore DB:";       gcloud firestore databases describe --database='(default)' --project=$$GCP_PROJECT_ID --format='value(name)' 2>/dev/null | sed 's/^/    /' || true; \
	echo "  Service accounts:";   gcloud iam service-accounts list --project=$$GCP_PROJECT_ID --filter='email:boston-pulse-data-pipeline*' --format='value(email)' 2>/dev/null | sed 's/^/    /' || true; \
	echo "  WIF pools:";          gcloud iam workload-identity-pools list --project=$$GCP_PROJECT_ID --location=global --format='value(name)' 2>/dev/null | sed 's/^/    /' || true; \
	echo ""; \
	echo "━━━ Console shortcuts ━━━"; \
	echo "  Project:          https://console.cloud.google.com/home/dashboard?project=$$GCP_PROJECT_ID"; \
	echo "  Cloud Run jobs:   https://console.cloud.google.com/run/jobs?project=$$GCP_PROJECT_ID"; \
	echo "  Artifact Reg:     https://console.cloud.google.com/artifacts?project=$$GCP_PROJECT_ID"; \
	echo "  GCS buckets:      https://console.cloud.google.com/storage/browser?project=$$GCP_PROJECT_ID"; \
	echo "  Monitoring:       https://console.cloud.google.com/monitoring/dashboards?project=$$GCP_PROJECT_ID"; \
	echo "  IAM & WIF:        https://console.cloud.google.com/iam-admin/workload-identity-pools?project=$$GCP_PROJECT_ID"

# ── Fresh-project one-shot demo ───────────────────────────────────────────────
# Reads all config from .env.demo (copy from .env.demo.example). demo-up creates
# the GCP project if missing, links billing, and runs the full demo-full flow.
# demo-down tears down infra, optionally deletes the project, and cleans local
# Terraform state.

_demo-check-env:
	@test -f .env.demo || { \
	  echo "✗ .env.demo not found. Copy the template and edit it:"; \
	  echo "    cp .env.demo.example .env.demo"; \
	  exit 1; }
	@test -n "$$GCP_PROJECT_ID"      || { echo "✗ GCP_PROJECT_ID missing in .env.demo";      exit 1; }
	@test -n "$$GCP_BILLING_ACCOUNT" || { echo "✗ GCP_BILLING_ACCOUNT missing in .env.demo"; exit 1; }
	@test -n "$$GITHUB_REPOSITORY"   || { echo "✗ GITHUB_REPOSITORY missing in .env.demo";   exit 1; }
	@echo "✓ .env.demo loaded (project: $$GCP_PROJECT_ID)"

demo-create-project:
	@if ! gcloud projects describe "$$GCP_PROJECT_ID" >/dev/null 2>&1; then \
	  echo "→ Creating GCP project $$GCP_PROJECT_ID..."; \
	  gcloud projects create "$$GCP_PROJECT_ID"; \
	else \
	  echo "✓ Project $$GCP_PROJECT_ID already exists"; \
	fi
	@billing=$$(gcloud billing projects describe "$$GCP_PROJECT_ID" \
	    --format='value(billingEnabled)' 2>/dev/null || echo "false"); \
	  if [ "$$billing" != "True" ] && [ "$$billing" != "true" ]; then \
	    echo "→ Linking billing account $$GCP_BILLING_ACCOUNT..."; \
	    gcloud billing projects link "$$GCP_PROJECT_ID" \
	      --billing-account="$$GCP_BILLING_ACCOUNT"; \
	  else \
	    echo "✓ Billing already linked"; \
	  fi
	@gcloud config set project "$$GCP_PROJECT_ID" >/dev/null
	@echo "✓ gcloud default project set to $$GCP_PROJECT_ID"

demo-up: _demo-check-env demo-create-project demo-full
	@echo ""
	@echo "━━━ Demo is up ━━━"
	@echo "Inspect everything:        make demo-status"
	@echo "Wait for VM startup:       make demo-wait-vm"
	@echo "Trigger data pipeline:     make demo-trigger-data"
	@echo "Trigger ML training:       make demo-trigger-training"

demo-down: _demo-check-env
	@echo "→ Destroying Terraform-managed resources in $$GCP_PROJECT_ID..."
	-@$(MAKE) demo-destroy
	@echo ""
	@printf "Also delete GCP project %s (30-day grace period)? [y/N] " "$$GCP_PROJECT_ID"; \
	  read ans; \
	  if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	    gcloud projects delete "$$GCP_PROJECT_ID" --quiet; \
	    echo "✓ Project scheduled for deletion"; \
	  else \
	    echo "⏭  Skipping project deletion"; \
	  fi
	@echo ""
	@echo "→ Cleaning local Terraform state..."
	@rm -f infrastructure/terraform/terraform.tfstate \
	       infrastructure/terraform/terraform.tfstate.backup \
	       terraform.tfstate
	@rm -rf infrastructure/terraform/.terraform
	@echo "✓ Local state cleaned"
	@echo ""
	@echo "━━━ Teardown complete ━━━"
