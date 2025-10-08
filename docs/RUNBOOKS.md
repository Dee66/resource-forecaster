# Runbooks (operational playbooks)

Concise, copy-paste friendly runbooks for common operational tasks. These are intended for an on-call engineer or SRE running the system in development or pre-prod. For production environments, follow your organization's deployment governance and the protected CI/CD pipeline.

## Preconditions (common)
- Ensure your AWS credentials are set (use an IAM profile with deploy permissions):
  - export AWS_PROFILE=your-deploy-profile
- Ensure environment variables used by the guardrails are present (e.g., BUDGET_DEV, BUDGET_STAGING, BUDGET_PROD) when running the CI guard steps locally.
- You should have the repository virtualenv active via Poetry:
  - poetry install
  - poetry shell

## Deploy (CDK)
Steps to synthesize and deploy the CDK stack locally (development):

1. Run tests and lint before deploy (fast smoke):

```bash
poetry install
poetry run pytest -q
poetry run nox -s lint
```

2. Package model artifacts (uploads or prepares artifact zip used by Lambda):

```bash
poetry run python scripts/package_forecaster_artifacts.py
```

3. Synthesize the CloudFormation template (verify changes):

```bash
poetry run cdk synth -a 'python infra/app.py'
```

4. Deploy the stack (CI should gate production deploys; pass --require-approval never for dev):

```bash
poetry run cdk deploy ForecasterStack --require-approval never
```

Rollback (model artifact):
- Use `scripts/rollback_model.py` to revert the model package or adjust the stack to point to a previous artifact. For complete stack rollback, consider `cdk deploy` to a prior app or `cdk destroy` only if necessary.

## Manual retrain (RMSE alarm response)
If the RMSE CloudWatch alarm fires and automation doesn't fully handle retraining, you can start the Step Functions execution manually:

```bash
python - <<'PY'
import boto3
client = boto3.client('stepfunctions')
resp = client.start_execution(stateMachineArn='arn:aws:states:...:stateMachine:ForecasterStateMachine', input='{}')
print(resp)
PY
```

Or inspect the retrain starter Lambda logs in CloudWatch to debug why automatic start may have failed.

## Invoke / Debug locally
- To run Lambda handlers locally, unit tests and integration stubs are provided. Example:

```bash
poetry run python -c "from lambda.retrain import handler as r; print(r.lambda_handler({'test': True}, None))"
```

- To simulate a full daily run locally use demo scripts under `demo/` (if present) or run the Step Function via the AWS console with test inputs.

## Audit Reconciliation (daily)
Purpose: verify that recommendations written to the audit DynamoDB table were implemented.

Run locally (quick check):

```bash
poetry run python - <<'PY'
from src.forecaster.audit_reconciler import reconcile_audit_with_resources
from botocore.session import Session
ddb = Session().create_client('dynamodb')
summary = reconcile_audit_with_resources(ddb, table_name='RightsizingAuditTable', resource_checker=None)
print(summary)
PY
```

Notes:
- The reconciler attempts to detect resource presence via heuristics (EC2/RDS/ASG). When `resource_type` is present on audit items, the reconciler uses it — prefer including `resource_type` when writing audit events.
- For production scale, use paginated scans or an incremental work-queue (SQS) to process large audit tables.

## FinOps reporting (ad-hoc)
- Export CloudWatch dashboards via the console or use `scripts/capture_dashboard_screenshots.py` to capture visuals for demos.
- The repo includes sample dashboards under `dashboards/` used during demos.

## Troubleshooting (common)
- Tests fail with ModuleNotFoundError for `aws_cdk` or `boto3`:
  - Run `poetry install` to ensure the virtualenv has dependencies.
- CDK circular dependency when adding IAM permissions:
  - Avoid attaching StartExecution to a role referenced by the state machine; create an isolated role for retrain starter Lambdas.
- Reconciler reports false positives/negatives:
  - Ensure audit items include `resource_type` or verify naming conventions for the target resources.

## Emergency teardown
- Prefer `scripts/teardown.py` for safe environment cleanup. Only use `cdk destroy ForecasterStack` in emergency cases, and be prepared to re-create dependent resources.

---

If you'd like, I can expand any section into a step-by-step playbook with exact CloudWatch console navigation, sample IAM policy snippets, or a checklist for on-call engineers.
# Runbooks

This document contains high-level runbooks for common operational procedures.

## Deploy (CDK)
- Preconditions:
  - Ensure `AWS_PROFILE` points to a deploy-capable IAM profile.
  - Ensure `BUDGET_*` environment variables set for guardrail checks where applicable.
- Steps:
  1. Build and run tests locally: `poetry install && poetry run pytest -q`
  2. Package model artifacts: `scripts/package_forecaster_artifacts.py` (or `poetry run python scripts/package_forecaster_artifacts.py`)
  3. Synthesize: `cdk synth -a 'python infra/app.py'` (or `poetry run cdk synth`)
  4. Deploy: `cdk deploy ForecasterStack --require-approval never` (CI should gate deploys in production)
- Rollback:
  - Use CDK to revert to prior stack or use the rollback script `scripts/rollback_model.py` for model artifacts.

## Invoke / Debug
- To run a daily forecast locally (invokes Step Functions via local runner):
  - Use `poetry run python demo/run_local_forecast.py` (if available) or run the Lambda handler entrypoints directly with sample events in `demo/samples/`.

## Audit Reconciliation (daily)
- Purpose: confirm recommendations were applied.
- Steps:
  1. Invoke the reconciliation Lambda (scheduled) or run locally:
     - `poetry run python -c "from src.forecaster.audit_reconciler import reconcile_audit_with_resources; ..."`
  2. The reconcilers emits a summary (implemented vs missing) and can publish to `alerts_topic` or an SNS topic for operations.

## RMSE Alarm Handling
- If RMSE alarm triggers:
  - Check the retrain starter Lambda logs and Step Function executions.
  - Verify data freshness and quality (S3 historical bucket exists and recent data present).
  - If required, run a manual retrain: trigger the retrain pipeline via Step Functions console or via `poetry run python -c "import boto3; boto3.client('stepfunctions').start_execution(...)"`.

## Emergency Teardown
- Use `cdk destroy ForecasterStack` with caution; prefer using the `scripts/teardown.py` helper.

***

For detailed playbooks (run commands, IAM profiles), see the team wiki / runbook repository.
