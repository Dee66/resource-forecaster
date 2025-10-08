Access Anomaly Detector - Delivery Checklist
<div align="left" style="margin:1rem 0;"> <strong>Status:</strong> <span>0% complete (0/75 items)</span> <div style="display:flex; align-items:center; gap:0.75rem; margin-top:0.35rem;"> <div style="flex:1; height:14px; background:#1f2933; border-radius:999px; overflow-hidden;"> <div style="width:0%; height:100%; background:linear-gradient(90deg, #10b981, #22d3ee);"></div> </div> <code style="background:#0f172a; color:#ecfeff; padding:0.1rem 0.55rem; border-radius:999px; font-weight:600;">0%</code> </div> </div>

1. Environment & Tooling 🛠️
- [ ] Confirm Python 3.11 toolchain installed locally

- [ ] Initialize Poetry project with NLP/NER specific dependencies (e.g., spaCy, Hugging Face)

- [ ] Define default Nox sessions (lint, tests, format, e2e_security, package)

- [ ] Document AWS CLI v2.27.50 requirement and restricted IAM profile setup.

- [ ] Document mandatory usage of AWS Secrets Manager for all model/endpoint credentials.

- Recommended additions & learnings from Resource Forecaster:
	- [ ] Add a `config/` folder with per-environment YAMLs (e.g., `dev.yml`, `prod.yml`) and a single source-of-truth config loader at `src/detector/config.py`.
	- [ ] Make CLI/deploy scripts default to dry-run and require an explicit opt-in (`--apply` or `ALLOW_AWS_DEPLOY=1`) to prevent accidental AWS changes.
	- [ ] Document Poetry/local `.venv` usage and reproducible dev setup (poetry install, activate .venv).

2. Project Scaffolding 🧱
- [ ] Finalize repo structure (infra/, src/detector/, src/enrichment/, data/, tests/)

- [ ] Populate README detailing the Security & Compliance Automation objective.

- Practical additions from implementation experience:
	- [ ] Add a top-level `config/` with example `dev.yml` and `prod.yml` templates and a README explaining fields required by `src/detector/config.py`.
	- [ ] Include a `scripts/package_model_artifacts.py` and a `scripts/deploy_detector.py` that are dry-run by default and accept an `--apply` switch.

3. Security Log Ingestion & Tagging 🗄️
- [ ] Define input sources (e.g., CloudTrail/VPC Flow Logs) and ingestion contract (EventBridge/SQS).

- [ ] Define target entities (IAM Role ARNs, IP Addresses, VPC IDs, KMS Key IDs) for NER training.

- [ ] Implement synthetic log generator (data/generate_logs.py) for reproducible training/testing.

- [ ] Define final compliance output schema (entity_id, entity_type, risk_score, log_id).

- [ ] Outline data retention and immutability policies for raw security logs.

- Implementation notes:
	- [ ] Apply S3 lifecycle policies for raw log buckets (infrequent access, glacier transition, long-term expiration) as part of infra.
	- [ ] Ensure bucket names and retention windows are parameterizable via the global config loader.

4. NER Model Development & Training 🏷️
- [ ] Implement src/train/ner_train.py for fine-tuning a small language model for entity extraction.

- [ ] Implement data preparation for NER tagging and sequence labeling.

- [ ] Define and implement the Anomaly Scoring logic (statistical deviation, rare entity combos).

- [ ] Capture training metrics (F1-score for NER, precision/recall for anomaly) to JSON/CloudWatch.

- [ ] Notebook placeholders: 01-Entity-Labeling, 02-Anomaly-Profiling.

- Practical additions:
	- [ ] Add packaging scripts that upload model artifacts to S3 in a predictable layout: `s3://<bucket>/model-packages/<env>/model_package-<ts>.zip` and expose `current.txt` alias management.
	- [ ] Provide a local inference container template (`scripts/detector_inference.py`) for packaging inside Fargate/Lambda/SageMaker.

5. Testing & Quality Gates ✅
- [ ] Unit tests for entity extraction accuracy and labeling consistency.

- [ ] Unit tests for Anomaly Scoring calculation.

- [ ] Integration test stub for the full log ingestion → scoring flow.

- [ ] Configure pytest + coverage thresholds.

- [ ] Define quality gate: NER F1-score ≥ Baseline for CI pass.

- Testing patterns learned from Resource Forecaster:
	- [ ] Use botocore Stubber to unit-test AWS client paths without network calls (S3, SageMaker, StepFunctions, SNS, DynamoDB).
	- [ ] Add CDK template unit tests (using aws_cdk.assertions.Template) to validate generated resources and Conditions without deploying.
	- [ ] Ensure CLI and deploy scripts run in-process during tests (so monkeypatching/stubbing works) rather than via subprocess when possible.
	- [ ] Add tests for dry-run behavior (no AWS calls by default) and an `apply` test mode that uses Stubber to simulate AWS responses.

6. Real-Time Enrichment Handler 🚨
- [ ] Define inference service API contract for log enrichment.

- [ ] Implement src/inference/enrichment_handler.py (log parsing, NER tagging, anomaly scoring).

- [ ] Ensure KMS encryption and secure loading for all model artifacts.

- [ ] Add structured logging for audit trail traceability.

- [ ] Package inference code for secure containerization (e.g., Fargate/Lambda inside a private VPC).

- Operational additions:
	- [ ] Add audit persistence for actionable recommendations (DynamoDB audit table) and a nightly reconciler Lambda that checks whether recommended actions were implemented.
	- [ ] Emit CloudWatch custom metrics (e.g., `AnomalyRate`, `ReconciliationMissingCount`) and create alarms that can trigger retrain/workflow Lambdas.

7. Infrastructure (CDK) 🏗️
- [ ] Flesh out infra/security_detector_stack.py (VPC resources, IAM roles, KMS keys).

- [ ] Define deployment using private subnets and VPC Endpoints only.

- [ ] Define service configuration (Fargate/Lambda) with required least-privilege IAM.

- [ ] Tag every resource (App, Env, CostCenter) for audit and FinOps.

- [ ] Add CloudWatch alarms for invocation errors and log volume anomalies.

- CDK notes & best-practices from Resource Forecaster:
	- [ ] Parameterize infra via the central config (allow passing `model_bucket_name`, `data_bucket_name`, and `vpc_id` into the stack constructor).
	- [ ] For optional/inert resources (e.g., SageMaker endpoints for model serving), add a CloudFormation `CfnParameter` + `CfnCondition` so those resources are synth-only unless explicitly enabled.
	- [ ] When using L1 (Cfn*) constructs, prefer the typed property classes (e.g., `ProductionVariantProperty`, `ServerlessConfigProperty`) instead of raw dicts to avoid JSII deserialization errors.
	- [ ] Provide CDK unit tests that assert Conditions are set (resources exist but are gated by the `EnableX` parameter).

8. Compliance Workflow Orchestration ⛓️
- [ ] Document the Step Functions log ingestion → scoring → alert flow.

- [ ] Provision the state machine via CDK to handle log ingestion, entity extraction, and final scoring.

- [ ] Integrate service endpoints to route the final anomaly/entity output to Security Hub and EventBridge.

- [ ] Implement retry logic and DLQs for handling failed log processing (preserving the audit trail).

- [ ] Add CloudWatch dashboards and alarms for state machine success/failure metrics.

- Workflow additions:
	- [ ] Wire a retrain trigger: create an alarm on the anomaly metric that publishes to SNS which triggers a lightweight retrain Lambda to start the Step Functions workflow.
	- [ ] Provide a reconciler job (nightly) that produces a missing-count metric and alarm tied to an alerts SNS topic.

9. Deployment & Operations 🔁
- [ ] Implement automated packaging script for model artifact upload.

- [ ] Script deployment flow (package → cdk deploy) with guardrails.

- [ ] Document rollback playbook (revert model package ARN, revert endpoint).

- [ ] Provide teardown automation (Makefile/Nox session).

- [ ] Capture CloudWatch dashboards screenshots for demo (showing event flow).

- Operational guardrails:
	- [ ] Deployment scripts must be dry-run by default and require explicit `--apply` or `ALLOW_AWS_DEPLOY=1` to perform AWS operations.
	- [ ] Include a budget/PaS guardrail script that evaluates predicted cost/fidelity and blocks deploys if thresholds are exceeded.
	- [ ] Provide unit tests for packaging/deploy scripts (run in-process and use stubbed clients).

10. CI/CD 🔄
- [ ] Create .github/workflows/ci.yml (lint + tests + cdk synth).

- [ ] Add a security artifact validation stage in CI (e.g., check entity list for accidental PII inclusion).

- [ ] Add .github/workflows/deploy.yml with manual approval and environment protection.

- CI best-practices:
	- [ ] Run `poetry install` and use the project's local `.venv` in CI.
	- [ ] Run `pytest` with collection and coverage thresholds; fail the build on regressions.
	- [ ] Add a synth step that runs `cdk synth --all --no-staging` and runs Template assertions to validate parameter presence and Conditions.
	- [ ] Keep all AWS-calls in unit tests stubbed; only explicit integration tests (in a protected pipeline) may use real AWS resources.

11. Senior Leader Mandates · Security & Audit 🔒
- [ ] Enforce VPC-only access for all compute resources (Fargate, Lambda, SageMaker).

- [ ] Enforce KMS Customer-Managed Keys (CMKs) for encrypting S3 artifacts and feature stores.

- [ ] Define IAM Permission Boundary for the service role enforcing least-privilege and mandatory tagging.

- [ ] Define Audit Path assurance: verify logs and outputs flow to Security Hub/CloudTrail without modification.

- [ ] Apply S3 lifecycle policies for raw log buckets to enforce long-term retention.

- [ ] Establish Policy-as-a-Service (PaS) guardrail that blocks CDK deploys missing KMS configuration.

- Additional governance items learned:
	- [ ] Provision an audit DynamoDB table and ensure all automated recommendations/events are persisted for reconciliation and compliance.
	- [ ] Create a reconciliation Lambda and a CloudWatch metric `ReconciliationMissingCount` with an alarm and SNS action.

12. Documentation & Interview Prep 📚
- [ ] Draft ADR: NER Strategy & Anomaly Scoring rationale.

- [ ] Write runbooks (deploy, invoke, rollback, Audit Trail Tracing).

- [ ] Prepare demo script + talking points emphasizing Compliance Automation and Data Security.

- [ ] Add FAQ section (KMS usage, VPC flow, event latency, False Positive strategy).
Access Anomaly Detector - Delivery Checklist

Status: 0% complete (0/75 items)

## 1. Environment & Tooling 🛠️
- [ ] Confirm Python 3.11 toolchain installed locally

- [ ] Initialize Poetry project with NLP/NER specific dependencies (e.g., spaCy, Hugging Face)

- [ ] Define default Nox sessions (lint, tests, format, e2e_security, package)

- [ ] Document AWS CLI v2.27.50 requirement and restricted IAM profile setup.

- [ ] Document mandatory usage of AWS Secrets Manager for all model/endpoint credentials.

**Recommended additions & learnings from Resource Forecaster:**

- [ ] Add a `config/` folder with per-environment YAMLs (e.g., `dev.yml`, `prod.yml`) and a single source-of-truth config loader at `src/detector/config.py`.
- [ ] Make CLI/deploy scripts default to dry-run and require an explicit opt-in (`--apply` or `ALLOW_AWS_DEPLOY=1`) to prevent accidental AWS changes.
- [ ] Document Poetry/local `.venv` usage and reproducible dev setup (poetry install, activate .venv).

## 2. Project Scaffolding 🧱
- [ ] Finalize repo structure (infra/, src/detector/, src/enrichment/, data/, tests/)

- [ ] Populate README detailing the Security & Compliance Automation objective.

**Practical additions from implementation experience:**

- [ ] Add a top-level `config/` with example `dev.yml` and `prod.yml` templates and a README explaining fields required by `src/detector/config.py`.
- [ ] Include a `scripts/package_model_artifacts.py` and a `scripts/deploy_detector.py` that are dry-run by default and accept an `--apply` switch.

## 3. Security Log Ingestion & Tagging 🗄️
- [ ] Define input sources (e.g., CloudTrail/VPC Flow Logs) and ingestion contract (EventBridge/SQS).

- [ ] Define target entities (IAM Role ARNs, IP Addresses, VPC IDs, KMS Key IDs) for NER training.

- [ ] Implement synthetic log generator (data/generate_logs.py) for reproducible training/testing.

- [ ] Define final compliance output schema (entity_id, entity_type, risk_score, log_id).

- [ ] Outline data retention and immutability policies for raw security logs.

**Implementation notes:**

- [ ] Apply S3 lifecycle policies for raw log buckets (infrequent access, glacier transition, long-term expiration) as part of infra.
- [ ] Ensure bucket names and retention windows are parameterizable via the global config loader.

## 4. NER Model Development & Training 🏷️
- [ ] Implement src/train/ner_train.py for fine-tuning a small language model for entity extraction.

- [ ] Implement data preparation for NER tagging and sequence labeling.

- [ ] Define and implement the Anomaly Scoring logic (statistical deviation, rare entity combos).

- [ ] Capture training metrics (F1-score for NER, precision/recall for anomaly) to JSON/CloudWatch.

- [ ] Notebook placeholders: 01-Entity-Labeling, 02-Anomaly-Profiling.

**Practical additions:**

- [ ] Add packaging scripts that upload model artifacts to S3 in a predictable layout: `s3://<bucket>/model-packages/<env>/model_package-<ts>.zip` and expose `current.txt` alias management.
- [ ] Provide a local inference container template (`scripts/detector_inference.py`) for packaging inside Fargate/Lambda/SageMaker.

## 5. Testing & Quality Gates ✅
- [ ] Unit tests for entity extraction accuracy and labeling consistency.

- [ ] Unit tests for Anomaly Scoring calculation.

- [ ] Integration test stub for the full log ingestion → scoring flow.

- [ ] Configure pytest + coverage thresholds.

- [ ] Define quality gate: NER F1-score ≥ Baseline for CI pass.

**Testing patterns learned from Resource Forecaster:**

- [ ] Use botocore Stubber to unit-test AWS client paths without network calls (S3, SageMaker, StepFunctions, SNS, DynamoDB).
- [ ] Add CDK template unit tests (using aws_cdk.assertions.Template) to validate generated resources and Conditions without deploying.
- [ ] Ensure CLI and deploy scripts run in-process during tests (so monkeypatching/stubbing works) rather than via subprocess when possible.
- [ ] Add tests for dry-run behavior (no AWS calls by default) and an `apply` test mode that uses Stubber to simulate AWS responses.

## 6. Real-Time Enrichment Handler 🚨
- [ ] Define inference service API contract for log enrichment.

- [ ] Implement src/inference/enrichment_handler.py (log parsing, NER tagging, anomaly scoring).

- [ ] Ensure KMS encryption and secure loading for all model artifacts.

- [ ] Add structured logging for audit trail traceability.

- [ ] Package inference code for secure containerization (e.g., Fargate/Lambda inside a private VPC).

**Operational additions:**

- [ ] Add audit persistence for actionable recommendations (DynamoDB audit table) and a nightly reconciler Lambda that checks whether recommended actions were implemented.
- [ ] Emit CloudWatch custom metrics (e.g., `AnomalyRate`, `ReconciliationMissingCount`) and create alarms that can trigger retrain/workflow Lambdas.

## 7. Infrastructure (CDK) 🏗️
- [ ] Flesh out infra/security_detector_stack.py (VPC resources, IAM roles, KMS keys).

- [ ] Define deployment using private subnets and VPC Endpoints only.

- [ ] Define service configuration (Fargate/Lambda) with required least-privilege IAM.

- [ ] Tag every resource (App, Env, CostCenter) for audit and FinOps.

- [ ] Add CloudWatch alarms for invocation errors and log volume anomalies.

**CDK notes & best-practices from Resource Forecaster:**

- [ ] Parameterize infra via the central config (allow passing `model_bucket_name`, `data_bucket_name`, and `vpc_id` into the stack constructor).
- [ ] For optional/inert resources (e.g., SageMaker endpoints for model serving), add a CloudFormation `CfnParameter` + `CfnCondition` so those resources are synth-only unless explicitly enabled.
- [ ] When using L1 (Cfn*) constructs, prefer the typed property classes (e.g., `ProductionVariantProperty`, `ServerlessConfigProperty`) instead of raw dicts to avoid JSII deserialization errors.
- [ ] Provide CDK unit tests that assert Conditions are set (resources exist but are gated by the `EnableX` parameter).

## 8. Compliance Workflow Orchestration ⛓️
- [ ] Document the Step Functions log ingestion → scoring → alert flow.

- [ ] Provision the state machine via CDK to handle log ingestion, entity extraction, and final scoring.

- [ ] Integrate service endpoints to route the final anomaly/entity output to Security Hub and EventBridge.

- [ ] Implement retry logic and DLQs for handling failed log processing (preserving the audit trail).

- [ ] Add CloudWatch dashboards and alarms for state machine success/failure metrics.

**Workflow additions:**

- [ ] Wire a retrain trigger: create an alarm on the anomaly metric that publishes to SNS which triggers a lightweight retrain Lambda to start the Step Functions workflow.
- [ ] Provide a reconciler job (nightly) that produces a missing-count metric and alarm tied to an alerts SNS topic.

## 9. Deployment & Operations 🔁
- [ ] Implement automated packaging script for model artifact upload.

- [ ] Script deployment flow (package → cdk deploy) with guardrails.

- [ ] Document rollback playbook (revert model package ARN, revert endpoint).

- [ ] Provide teardown automation (Makefile/Nox session).

- [ ] Capture CloudWatch dashboards screenshots for demo (showing event flow).

**Operational guardrails:**

- [ ] Deployment scripts must be dry-run by default and require explicit `--apply` or `ALLOW_AWS_DEPLOY=1` to perform AWS operations.
- [ ] Include a budget/PaS guardrail script that evaluates predicted cost/fidelity and blocks deploys if thresholds are exceeded.
- [ ] Provide unit tests for packaging/deploy scripts (run in-process and use stubbed clients).

## 10. CI/CD 🔄
- [ ] Create .github/workflows/ci.yml (lint + tests + cdk synth).

- [ ] Add a security artifact validation stage in CI (e.g., check entity list for accidental PII inclusion).

- [ ] Add .github/workflows/deploy.yml with manual approval and environment protection.

**CI best-practices:**

- [ ] Run `poetry install` and use the project's local `.venv` in CI.
- [ ] Run `pytest` with collection and coverage thresholds; fail the build on regressions.
- [ ] Add a synth step that runs `cdk synth --all --no-staging` and runs Template assertions to validate parameter presence and Conditions.
- [ ] Keep all AWS-calls in unit tests stubbed; only explicit integration tests (in a protected pipeline) may use real AWS resources.

## 11. Senior Leader Mandates · Security & Audit 🔒
- [ ] Enforce VPC-only access for all compute resources (Fargate, Lambda, SageMaker).

- [ ] Enforce KMS Customer-Managed Keys (CMKs) for encrypting S3 artifacts and feature stores.

- [ ] Define IAM Permission Boundary for the service role enforcing least-privilege and mandatory tagging.

- [ ] Define Audit Path assurance: verify logs and outputs flow to Security Hub/CloudTrail without modification.

- [ ] Apply S3 lifecycle policies for raw log buckets to enforce long-term retention.

- [ ] Establish Policy-as-a-Service (PaS) guardrail that blocks CDK deploys missing KMS configuration.

**Additional governance items learned:**

- [ ] Provision an audit DynamoDB table and ensure all automated recommendations/events are persisted for reconciliation and compliance.
- [ ] Create a reconciliation Lambda and a CloudWatch metric `ReconciliationMissingCount` with an alarm and SNS action.

## 12. Documentation & Interview Prep 📚
- [ ] Draft ADR: NER Strategy & Anomaly Scoring rationale.

- [ ] Write runbooks (deploy, invoke, rollback, Audit Trail Tracing).

- [ ] Prepare demo script + talking points emphasizing Compliance Automation and Data Security.

- [ ] Add FAQ section (KMS usage, VPC flow, event latency, False Positive strategy).
Access Anomaly Detector - Delivery Checklist

Status: 0% complete (0/75 items)

## 1. Environment & Tooling 🛠️
- [ ] Confirm Python 3.11 toolchain installed locally

- [ ] Initialize Poetry project with NLP/NER specific dependencies (e.g., spaCy, Hugging Face)

- [ ] Define default Nox sessions (lint, tests, format, e2e_security, package)

- [ ] Document AWS CLI v2.27.50 requirement and restricted IAM profile setup.

- [ ] Document mandatory usage of AWS Secrets Manager for all model/endpoint credentials.

**Recommended additions & learnings from Resource Forecaster:**

- [ ] Add a `config/` folder with per-environment YAMLs (e.g., `dev.yml`, `prod.yml`) and a single source-of-truth config loader at `src/detector/config.py`.
- [ ] Make CLI/deploy scripts default to dry-run and require an explicit opt-in (`--apply` or `ALLOW_AWS_DEPLOY=1`) to prevent accidental AWS changes.
- [ ] Document Poetry/local `.venv` usage and reproducible dev setup (poetry install, activate .venv).

## 2. Project Scaffolding 🧱
- [ ] Finalize repo structure (infra/, src/detector/, src/enrichment/, data/, tests/)

- [ ] Populate README detailing the Security & Compliance Automation objective.

**Practical additions from implementation experience:**

- [ ] Add a top-level `config/` with example `dev.yml` and `prod.yml` templates and a README explaining fields required by `src/detector/config.py`.
- [ ] Include a `scripts/package_model_artifacts.py` and a `scripts/deploy_detector.py` that are dry-run by default and accept an `--apply` switch.

## 3. Security Log Ingestion & Tagging 🗄️
- [ ] Define input sources (e.g., CloudTrail/VPC Flow Logs) and ingestion contract (EventBridge/SQS).

- [ ] Define target entities (IAM Role ARNs, IP Addresses, VPC IDs, KMS Key IDs) for NER training.

- [ ] Implement synthetic log generator (data/generate_logs.py) for reproducible training/testing.

- [ ] Define final compliance output schema (entity_id, entity_type, risk_score, log_id).

- [ ] Outline data retention and immutability policies for raw security logs.

**Implementation notes:**

- [ ] Apply S3 lifecycle policies for raw log buckets (infrequent access, glacier transition, long-term expiration) as part of infra.
- [ ] Ensure bucket names and retention windows are parameterizable via the global config loader.

## 4. NER Model Development & Training 🏷️
- [ ] Implement src/train/ner_train.py for fine-tuning a small language model for entity extraction.

- [ ] Implement data preparation for NER tagging and sequence labeling.

- [ ] Define and implement the Anomaly Scoring logic (statistical deviation, rare entity combos).

- [ ] Capture training metrics (F1-score for NER, precision/recall for anomaly) to JSON/CloudWatch.

- [ ] Notebook placeholders: 01-Entity-Labeling, 02-Anomaly-Profiling.

**Practical additions:**

- [ ] Add packaging scripts that upload model artifacts to S3 in a predictable layout: `s3://<bucket>/model-packages/<env>/model_package-<ts>.zip` and expose `current.txt` alias management.
- [ ] Provide a local inference container template (`scripts/detector_inference.py`) for packaging inside Fargate/Lambda/SageMaker.

## 5. Testing & Quality Gates ✅
- [ ] Unit tests for entity extraction accuracy and labeling consistency.

- [ ] Unit tests for Anomaly Scoring calculation.

- [ ] Integration test stub for the full log ingestion → scoring flow.

- [ ] Configure pytest + coverage thresholds.

- [ ] Define quality gate: NER F1-score ≥ Baseline for CI pass.

**Testing patterns learned from Resource Forecaster:**

- [ ] Use botocore Stubber to unit-test AWS client paths without network calls (S3, SageMaker, StepFunctions, SNS, DynamoDB).
- [ ] Add CDK template unit tests (using aws_cdk.assertions.Template) to validate generated resources and Conditions without deploying.
- [ ] Ensure CLI and deploy scripts run in-process during tests (so monkeypatching/stubbing works) rather than via subprocess when possible.
- [ ] Add tests for dry-run behavior (no AWS calls by default) and an `apply` test mode that uses Stubber to simulate AWS responses.

## 6. Real-Time Enrichment Handler 🚨
- [ ] Define inference service API contract for log enrichment.

- [ ] Implement src/inference/enrichment_handler.py (log parsing, NER tagging, anomaly scoring).

- [ ] Ensure KMS encryption and secure loading for all model artifacts.

- [ ] Add structured logging for audit trail traceability.

- [ ] Package inference code for secure containerization (e.g., Fargate/Lambda inside a private VPC).

**Operational additions:**

- [ ] Add audit persistence for actionable recommendations (DynamoDB audit table) and a nightly reconciler Lambda that checks whether recommended actions were implemented.
- [ ] Emit CloudWatch custom metrics (e.g., `AnomalyRate`, `ReconciliationMissingCount`) and create alarms that can trigger retrain/workflow Lambdas.

## 7. Infrastructure (CDK) 🏗️
- [ ] Flesh out infra/security_detector_stack.py (VPC resources, IAM roles, KMS keys).

- [ ] Define deployment using private subnets and VPC Endpoints only.

- [ ] Define service configuration (Fargate/Lambda) with required least-privilege IAM.

- [ ] Tag every resource (App, Env, CostCenter) for audit and FinOps.

- [ ] Add CloudWatch alarms for invocation errors and log volume anomalies.

**CDK notes & best-practices from Resource Forecaster:**

- [ ] Parameterize infra via the central config (allow passing `model_bucket_name`, `data_bucket_name`, and `vpc_id` into the stack constructor).
- [ ] For optional/inert resources (e.g., SageMaker endpoints for model serving), add a CloudFormation `CfnParameter` + `CfnCondition` so those resources are synth-only unless explicitly enabled.
- [ ] When using L1 (Cfn*) constructs, prefer the typed property classes (e.g., `ProductionVariantProperty`, `ServerlessConfigProperty`) instead of raw dicts to avoid JSII deserialization errors.
- [ ] Provide CDK unit tests that assert Conditions are set (resources exist but are gated by the `EnableX` parameter).

## 8. Compliance Workflow Orchestration ⛓️
- [ ] Document the Step Functions log ingestion → scoring → alert flow.

- [ ] Provision the state machine via CDK to handle log ingestion, entity extraction, and final scoring.

- [ ] Integrate service endpoints to route the final anomaly/entity output to Security Hub and EventBridge.

- [ ] Implement retry logic and DLQs for handling failed log processing (preserving the audit trail).

- [ ] Add CloudWatch dashboards and alarms for state machine success/failure metrics.

**Workflow additions:**

- [ ] Wire a retrain trigger: create an alarm on the anomaly metric that publishes to SNS which triggers a lightweight retrain Lambda to start the Step Functions workflow.
- [ ] Provide a reconciler job (nightly) that produces a missing-count metric and alarm tied to an alerts SNS topic.

## 9. Deployment & Operations 🔁
- [ ] Implement automated packaging script for model artifact upload.

- [ ] Script deployment flow (package → cdk deploy) with guardrails.

- [ ] Document rollback playbook (revert model package ARN, revert endpoint).

- [ ] Provide teardown automation (Makefile/Nox session).

- [ ] Capture CloudWatch dashboards screenshots for demo (showing event flow).

**Operational guardrails:**

- [ ] Deployment scripts must be dry-run by default and require explicit `--apply` or `ALLOW_AWS_DEPLOY=1` to perform AWS operations.
- [ ] Include a budget/PaS guardrail script that evaluates predicted cost/fidelity and blocks deploys if thresholds are exceeded.
- [ ] Provide unit tests for packaging/deploy scripts (run in-process and use stubbed clients).

## 10. CI/CD 🔄
- [ ] Create .github/workflows/ci.yml (lint + tests + cdk synth).

- [ ] Add a security artifact validation stage in CI (e.g., check entity list for accidental PII inclusion).

- [ ] Add .github/workflows/deploy.yml with manual approval and environment protection.

**CI best-practices:**

- [ ] Run `poetry install` and use the project's local `.venv` in CI.
- [ ] Run `pytest` with collection and coverage thresholds; fail the build on regressions.
- [ ] Add a synth step that runs `cdk synth --all --no-staging` and runs Template assertions to validate parameter presence and Conditions.
- [ ] Keep all AWS-calls in unit tests stubbed; only explicit integration tests (in a protected pipeline) may use real AWS resources.

## 11. Senior Leader Mandates · Security & Audit 🔒
- [ ] Enforce VPC-only access for all compute resources (Fargate, Lambda, SageMaker).

- [ ] Enforce KMS Customer-Managed Keys (CMKs) for encrypting S3 artifacts and feature stores.

- [ ] Define IAM Permission Boundary for the service role enforcing least-privilege and mandatory tagging.

- [ ] Define Audit Path assurance: verify logs and outputs flow to Security Hub/CloudTrail without modification.

- [ ] Apply S3 lifecycle policies for raw log buckets to enforce long-term retention.

- [ ] Establish Policy-as-a-Service (PaS) guardrail that blocks CDK deploys missing KMS configuration.

**Additional governance items learned:**

- [ ] Provision an audit DynamoDB table and ensure all automated recommendations/events are persisted for reconciliation and compliance.
- [ ] Create a reconciliation Lambda and a CloudWatch metric `ReconciliationMissingCount` with an alarm and SNS action.

## 12. Documentation & Interview Prep 📚
- [ ] Draft ADR: NER Strategy & Anomaly Scoring rationale.

- [ ] Write runbooks (deploy, invoke, rollback, Audit Trail Tracing).

- [ ] Prepare demo script + talking points emphasizing Compliance Automation and Data Security.

- [ ] Add FAQ section (KMS usage, VPC flow, event latency, False Positive strategy).

- [ ] Capture lessons learned / future enhancements section.

## Supplement — Lessons & recommendations from implementing resource-forecaster

The Resource Forecaster project produced a number of practical learnings that are directly applicable when building the `anomaly-detector` plugin. Below is a compact, prioritized set of recommendations, anti-patterns to avoid, and starter tasks you can pick up immediately.

Key lessons (high level)
- Use feature-flagged infra for optional, costly resources. Gate heavy resources (SageMaker endpoints, training jobs) behind CloudFormation parameters + Conditions so synth is safe and deploys are explicit.
- Prefer small, focused unit tests and CDK template assertions over integration runs in CI. Stub AWS calls (botocore Stubber) and test behavior in-process.
- Enforce dry-run by default across packaging and deploy scripts. Make any destructive action require an explicit `--apply` or environment flag (`ALLOW_AWS_DEPLOY=1`).
- Centralize configuration (one loader) and keep per-environment YAMLs under `config/` to avoid duplication and accidental prod changes.
- Tag everything consistently using a small set of canonical tag keys (Application, Environment, CostCenter). Treat tags as policy primitives (used by Secrets Manager queries, IAM conditions, and billing reports).

Concrete infra & repo patterns
- CDK stack constructor should accept explicit resource names or prefixes (e.g., `model_bucket_name`, `alerts_topic_name`) rather than composing them ad-hoc inside the stack. This makes testing and local synth deterministic.
- When creating SNS topics/alarms/dashboards, put human-readable DisplayName separate from the resource logical name. Tests can assert both.
- Use typed CDK constructs where available (avoid passing raw dicts to L1 constructs unless necessary). This reduces JSII serialization errors.
- Add a lightweight `cdk-unit-tests/` pattern: small pytest files that create a Stack in several input permutations and assert Template conditions.

Testing & quality gates (recommended)
- Unit tests: data processing and anomaly scoring (happy path + edge cases). For example: empty inputs, constant series, single outlier, multi-outliers, time-windowed scoring.
- CDK tests: Template assertions for required resources (buckets, IAM roles, KMS keys) and for gating parameters.
- Integration-test stage (optional, gated): runs in a protected CI environment using curated test accounts and real resources; isolate costs and run cleanup.
- CI: fail fast on collection or template assertion failures. Run `poetry install` inside the subproject and `pytest --maxfail=1 -q`.

Operational and observability recommendations
- Emit custom metrics for anomaly-rate and reconciliation-missing-count. Build alarms that auto-mute during deploy windows to avoid noise.
- Persist audit entries for decisions into DynamoDB with a reconciliation job that compares recommended vs. applied actions. Emit a metric for missing reconciliations.
- Provide dashboards and a script to snapshot dashboards (for demos) and store them under `dashboards/`.

Starter tasks for the anomaly-detector plugin (concrete, high-value)
1. Add `config/{dev,prod}.yml` and a `src/detector/config.py` loader that prefers env vars then file.
2. Create `infra/security_detector_stack.py` with inputs: `enable_training`, `model_bucket_name`, `alerts_topic_name`, `kms_key_arn` and template unit-tests (aws_cdk.assertions.Template-based).
3. Implement `src/detector/scoring.py` with a test suite covering modified z-score, windowed scoring, and a simple reconciliation stub.
4. Add `scripts/package_model_artifacts.py` with a dry-run flag and deterministic S3 layout `model-packages/{env}/...`.
5. Add CI job `anomaly-detector/ci.yml` to run lint, pytest and `cdk synth` for the plugin in isolation.

Quick commands (dev flow)
```bash
# inside repository root
cd anomaly-detector
poetry install
poetry run pytest -q
cd ../infra
# synth plugin stack (example)
cdk synth -a "python infra/app.py" --profile dev
```

Anti-patterns observed
- Making infra resource names purely derived from display strings. Keep a separation so tests and ARNs remain stable.
- Running integration-style CDK tests in the default CI without cost controls — this caused noise and failures during earlier runs.

Where this helps the next plugin
- The checklist items above already mirror many Resource Forecaster lessons; the supplement turns those practical tips into explicit starter tasks and patterns so you can iterate quickly when implementing the plugin.

If you want, I can now:
- Apply the top 5 starter tasks (create `config/` files, a minimal `infra/security_detector_stack.py` skeleton and template tests, and add CI workflow) as a follow-up.
- Produce example CDK template assertions for the SNS/KMS/bucket resources that the tests expect.
