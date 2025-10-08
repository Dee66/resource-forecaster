# Section 11: Senior Leader Mandates - FinOps & Capacity Runbook

This runbook implements the Senior Leader Mandates requested in Section 11. It
provides templates, automation scripts, and operational guidance for alarms,
non-prod shutdown, budget guardrails, rightsizing playbook assurance, and S3
lifecycle policies.

## Artifacts created
- `scripts/section11_create_alarms_and_guardrails.py`: Generate CloudWatch alarm
  JSON for RMSE drift and optionally create the alarm and SNS topic in AWS.
- `scripts/section11_budget_guardrail.py`: Compare predicted costs to a budget
  envelope and optionally block deploys (exit non-zero).
- `scripts/lambda_shutdown_nonprod.py`: Template Lambda handler to shutdown
  non-production resources based on recommendations.
- `dashboards/predicted_cost.json`: Sample predicted cost metadata used by guardrail checks.

## Alarms: RMSE Drift
- Alarm Name: `ResourceForecaster-RMSE-Drift-<env>`
- Metric: `ResourceForecaster/Metrics` → `RMSE` (Dimension: Environment)
- Threshold: default 0.05 (5% RMSE) over 3 evaluation periods (5m each)
- Action: SNS topic `ResourceForecaster-Alerts` for on-call notification

Create alarm definitions locally:

```bash
nox -s section11
```

Apply alarms and subscribe an email (requires AWS credentials):

```bash
nox -s section11 -- --apply --sns-email ops@example.com --env prod
```

## Budget Guardrail
- A pre-deploy guard checks `dashboards/predicted_cost.json` for predicted monthly
  costs by environment and compares them against a configured budget.
- To run the guard and block deploys when predicted cost exceeds budget:

```bash
nox -s section11 -- --check-budget --predicted-json dashboards/predicted_cost.json --budget 900 --block-exit --env prod
```

In CI or CDK pipeline, add the above command as a pre-deploy step. A non-zero exit
will block the pipeline.

## Non-Prod Shutdown Lambda
- The `lambda_shutdown_nonprod.py` file is a safe template showing how resource
  ARNs can be stopped. Implementers must wire exact IAM permissions:
  - ec2:StopInstances, rds:StopDBInstance, ecs:UpdateService (scale down), etc.
- Deploy the Lambda with a secure deployment pipeline and an approval step before
  executing mass shutdowns.

## Rightsizing Playbook Assurance
- Ensure every rightsizing recommendation is logged to a durable store (S3/DB).
- Periodically reconcile recommendations against actual changes (audit job).
- For critical recommendations, require manual approval workflow (e.g., via
  Step Functions or Slack approval integration).

## S3 Lifecycle Policies
- Apply lifecycle rules to historical data buckets to move older data to
  Glacier/Deep Archive after 90/365 days and expire after 3 years.
- Add lifecycle policy examples to infra/CDK when provisioning S3 buckets.

## Next steps
1. Wire CDK resource constructs to create CloudWatch alarms and SNS topics.
2. Add IAM least-privilege roles for the shutdown Lambda.
3. Wire the budget guardrail into the CDK deploy action in GitHub Actions to block deploys.
4. Add automated audit job verifying rightsizing recommendations are actioned.

