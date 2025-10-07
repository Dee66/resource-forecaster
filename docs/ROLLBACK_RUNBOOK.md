# Rollback Runbook – Resource Forecaster

This runbook describes how to roll back the active model/inference package to a previous version.

## Preconditions
- AWS credentials with s3:GetObject and s3:PutObject on the artifacts bucket
- Artifacts stored at: `s3://<bucket>/model-packages/<env>/model_package-<timestamp>.zip`
- Optional alias pointer at: `s3://<bucket>/model-packages/<env>/current.txt` containing the selected package key

## Quick commands (via Nox)
- List packages for an environment:
  - `nox -s rollback -- --env staging --list`
- Roll back to previous package (auto-select index 1), set alias, and download locally:
  - `nox -s rollback -- --env staging --previous --set-alias --download`
- Set alias to a specific key:
  - `nox -s rollback -- --env prod --key model-packages/prod/model_package-20250101-000000.zip --set-alias`

## Detailed steps
1) Identify candidate packages
   - Use `--list` to see available artifacts (newest first) and confirm timestamps.
2) Select a target version
   - Either pass `--key` explicitly or `--previous` to auto-select the second newest package.
3) Update the alias pointer
   - Use `--set-alias` to overwrite `current.txt`. Downstream deployment should read this alias to fetch the correct package.
4) Verify
   - Validate the alias content and optionally re-deploy the service/environment to pick up the reverted package.

## Notes
- If your deploy pipeline reads `current.txt`, a re-deploy or rolling restart will propagate the rollback.
- For immutable deployments, trigger a deployment with the selected package key instead of alias.
- Keep a retention policy so older model packages remain available for rollbacks.
