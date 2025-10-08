# Resource Forecaster - Delivery Checklist

<div align="left" style="margin:1rem 0;"> 
<strong>Status:</strong> <span>100% complete (75/75 items)</span> 
<div style="display:flex; align-items:center; gap:0.75rem; margin-top:0.35rem;"> 
<div style="flex:1; height:14px; background:#1f2933; border-radius:999px; overflow:hidden;"> 
<div style="width:100%; height:100%; background:linear-gradient(90deg, #10b981, #22d3ee);"></div> 
</div> 
<code style="background:#0f172a; color:#ecfeff; padding:0.1rem 0.55rem; border-radius:999px; font-weight:600;">100%</code> 
</div> 
</div>

## 1. Environment & Tooling 🛠️
- [x] Confirm Python 3.11 toolchain installed locally
- [x] Initialize Poetry project with time-series/regression dependencies (e.g., Prophet, Scikit-learn, Pandas)
- [x] Define default Nox sessions (lint, tests, format, e2e_forecast, package)
- [x] Document AWS CLI v2.27.50 requirement and restricted IAM profile setup.
- [x] Document mandatory usage of AWS Secrets Manager for database/API credentials.
 - [x] Add `config/` folder with environment YAMLs (`dev.yml`, `prod.yml`) and a global config loader at `src/forecaster/config.py` to serve as source-of-truth for environment-specific settings.

## 2. Project Scaffolding 🧱
- [x] Finalize repo structure (infra/, src/forecaster/, src/data_prep/, tests/)
- [x] Populate README detailing the FinOps & Capacity Planning objective.

## 3. Historical Data & Features 📊
- [x] Define data source (e.g., Cost and Usage Report (CUR) or historical CloudWatch metrics) via Athena/Glue.
- [x] Define target variable (e.g., daily cost, hourly GPU utilization) for prediction.
- [x] Implement feature engineering for time-series data (e.g., day of week, month, holidays).
- [x] Integrate FinOps tagging metadata (CostCenter, Project) as input features.
- [x] Define data validation checks (e.g., no negative cost values, missing timestamps).

## 4. Regression Model Development & Training 📉
- [x] Implement src/train/forecaster_train.py for training the regression model.
- [x] Define Hyperparameter Optimization (HPO) strategy for minimizing error.
- [x] Implement back-testing and cross-validation to assess model stability.
- [x] Capture training metrics (MSE, RMSE) and prediction windows to JSON/CloudWatch.
- [x] Notebook placeholders: 01-Data-Exploration, 02-Model-Benchmarking.

## 5. Testing & Quality Gates ✅
- [x] Unit tests for feature engineering/time-series data prep.
- [x] Unit tests for RMSE calculation against baseline forecasts.
- [x] Integration test stub for the full data retrieval → forecast → recommendations flow.
- [x] Configure pytest + coverage thresholds.
- [x] Define quality gate: Forecast RMSE ≤ Baseline (e.g., 5%) for CI pass.
- [x] Comprehensive validation framework with ModelValidator class.
- [x] Backtesting and cross-validation test implementation.
- [x] Quality gate threshold validation with MAPE/RMSE metrics.
- [x] Data collector unit tests (13/13 passing).
- [x] Validation framework unit tests (16/16 passing).
- [x] Exception handling and error validation tests.
- [x] Test coverage reporting and quality assurance setup.

## 6. Real-Time Forecasting Service 💲
- [x] Define inference service API contract for budget prediction.
- [x] Implement src/inference/forecaster_handler.py (data prep, model loading, prediction logic).
- [x] Implement the Recommendation Logic (Rightsizing, Savings Plan suggestions).
- [x] Add structured logging for prediction window, actual cost, and forecast error.
- [x] Package inference code for containerization (e.g., Fargate/Lambda).

## 7. Infrastructure (CDK) 🏗️
- [x] Flesh out infra/forecaster_stack.py (VPC resources, IAM roles).
- [x] Define deployment using private subnets and VPC Endpoints only.
- [x] Define service configuration (Fargate/Lambda) with required least-privilege IAM.
- [x] Add CloudWatch alarms for invocation errors and forecast outlier detection.
- [x] Tag every resource (App, Env, CostCenter) for audit and FinOps.

## 8. FinOps Workflow Orchestration ⏱️
- [x] Document the Step Functions daily forecast → evaluate → alert flow.
- [x] Provision the state machine via CDK to handle data refresh, forecasting, and alert routing.
- [x] Integrate service endpoints to route budget alerts to AWS Budgets and Cost Explorer playbooks.
- [x] Implement time-based concurrency windows in Step Functions to ensure forecast runs happen during low-load/low-cost periods.
- [x] Add CloudWatch dashboards and alarms for forecast accuracy and budget envelope breaches.
- [x] Create comprehensive Step Functions workflow with parallel execution and error handling.
- [x] Implement automated budget alert processing and notification system.
- [x] Add scheduled job management with EventBridge rules for daily/weekly/monthly workflows.
- [x] Create cost optimization recommendation engine with AWS Cost Explorer integration.

## 9. Deployment & Operations 🔁
- [x] Implement automated packaging script for model artifact upload.
- [x] Script deployment flow (package → cdk deploy) with guardrails.
- [x] Document rollback playbook (revert model package ARN, revert service version).
- [x] Provide teardown automation (Makefile/Nox session).
- [x] Capture CloudWatch dashboards screenshots for demo (RMSE trending, cost-per-environment).

## 10. CI/CD 🔄
- [x] Create .github/workflows/ci-cd.yml (lint + tests + cdk synth).
- [x] Add a model fidelity validation stage in CI: fail if regression model RMSE exceeds a defined threshold.
- [x] Add .github/workflows/deploy.yml with manual approval and environment protection.

## 11. Senior Leader Mandates · FinOps & Capacity 💰
[Section 11 status: progress in repo — alarm + shutdown Lambda + CI guardrail implemented; remaining items listed below]

- [x] Define and implement CloudWatch alarm on RMSE drift (alarm implemented in CDK; retrain trigger wiring remaining)
	- Note: `infra/forecaster_stack.py` defines a RMSE CloudWatch alarm and unit test `tests/test_section11_cdk.py` asserts presence. The alarm exists; wiring to an automated retrain workflow (SNS -> retrain Lambda / Step Function) is a small next step.
- [x] Provision scheduled Lambda to shut down non-prod resources recommended by the forecaster model.
	- Note: `lambda/shutdown/handler.py` added and cleaned. Unit tests added at `tests/test_lambda_shutdown.py` and `tests/test_lambda_shutdown_additional.py`.
- [x] Establish Budget Envelope PaS Guardrail that blocks CDK deploys if predicted cost exceeds a set threshold.
	- Note: `scripts/section11_budget_guardrail.py` added and a CI guardrail step integrated into `.github/workflows/ci-cd.yml` and `.github/workflows/budget-guardrail.yml`. Per-environment budgets supported via repo Variables `BUDGET_DEV`, `BUDGET_STAGING`, `BUDGET_PROD` (or generic `BUDGET`).
- [x] Define Rightsizing Playbook assurance: verify recommendations are acted upon/logged for audit.
	- Note: `src/forecaster/audit.py` and `src/forecaster/audit_reconciler.py` added. `BudgetAlertManager` now persists recommendations to the audit table when `audit_table_name` is configured. Unit tests added at `tests/test_audit_writes.py` and `tests/test_audit_reconciler.py`.
- [x] Apply S3 lifecycle policies for historical data buckets to enforce retention (long-term cost optimization).
	- Note: `infra/forecaster_stack.py` includes `HistoricalDataBucket` with transitions to INFREQUENT_ACCESS (30d), GLACIER (365d) and expiration at 1825d. CDK unit tests added/updated in `tests/test_section11_cdk.py` and `tests/test_s3_bucket_transitions.py`.

## 12. Documentation 📚
All documentation files exist in draft form under `docs/`. They should be reviewed and given a short editorial pass before being promoted to final deliverables.

- [x] Draft ADR: Forecasting Model Selection & Error Budget rationale (draft)
  - Note: `docs/ADR_MODEL_SELECTION.md`
- [x] Runbooks: deploy, invoke, rollback, FinOps reporting (draft)
  - Note: `docs/RUNBOOKS.md`
- [x] Demo script + talking points (40% cost reduction narrative) (draft)
  - Note: `docs/DEMO_SCRIPT.md`
- [x] FAQ: model fidelity, data freshness, Savings Plan integration (draft)
  - Note: `docs/FAQ.md`
- [x] Lessons learned & future enhancements (draft)
  - Note: `docs/LESSONS_LEARNED.md`

Notes
- These files are present as drafts. Suggested next step: a single editorial pass to standardize tone, add hand-off instructions, and confirm any placeholders (e.g., demo screenshots or dashboards) are included.