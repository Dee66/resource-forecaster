#!/usr/bin/env python3
"""Deployment flow: package -> synth -> deploy with guardrails.

Guardrails:
- Prevent prod deploys from non-main/master unless --force
- Require confirmation for prod unless --yes
- Optionally upload package to S3
- Optional smoke tests post-deploy
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check).returncode


def get_branch() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return os.getenv("GITHUB_REF_NAME", "")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Package and deploy with CDK")
    p.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Target environment")
    p.add_argument("--s3-bucket", default=os.getenv("S3_MODEL_ARTIFACTS_BUCKET"), help="Optional S3 bucket for artifacts upload")
    p.add_argument("--yes", action="store_true", help="Auto-confirm prompts (required for prod)")
    p.add_argument("--force", action="store_true", help="Bypass branch guardrails for prod")
    p.add_argument("--allow-dirty", action="store_true", help="Skip clean working tree check")
    p.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests after deployment")
    args = p.parse_args(argv)

    # Guardrails
    branch = get_branch()
    if args.env == "prod" and not args.force:
        if branch not in {"main", "master"}:
            print(f"❌ Refusing to deploy prod from branch '{branch}'. Use --force to override.")
            return 2
        if not args.yes:
            print("❌ Production deploy requires --yes confirmation flag.")
            return 2

    if not args.allow_dirty:
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode().strip()
            if status:
                print("❌ Working tree has uncommitted changes. Commit or use --allow-dirty.")
                return 2
        except Exception:
            pass

    # If s3-bucket wasn't passed, prefer repo config (post-parse)
    if not args.s3_bucket:
        try:
            from src.forecaster.config import load_config

            cfg = load_config(args.env)
            args.s3_bucket = cfg.infrastructure.model_bucket
        except Exception:
            pass

    # Step 1: Package artifacts (upload if bucket provided)
    pkg_cmd = [sys.executable, str(ROOT / "scripts" / "package_forecaster_artifacts.py"), "--env", args.env]
    if args.s3_bucket:
        pkg_cmd += ["--s3-bucket", args.s3_bucket]
    run(["poetry", "run", *pkg_cmd], cwd=ROOT)

    # Step 2: CDK synth
    infra_dir = ROOT / "infra"
    run(["poetry", "run", "cdk", "synth", "--all", "--no-staging"], cwd=infra_dir)

    # Step 3: CDK deploy
    suffix = args.env.capitalize()
    stack = f"ResourceForecaster-{suffix}"
    deploy_cmd = [
        "poetry", "run", "cdk", "deploy", stack,
        "--require-approval", "never",
        "--context", f"environment={args.env}",
    ]
    run(deploy_cmd, cwd=infra_dir)

    # Step 4: Smoke tests (optional)
    if not args.skip_smoke and (ROOT / "tests" / "smoke").exists():
        run(["poetry", "run", "pytest", "-q", "tests/smoke"], cwd=ROOT, check=False)

    print(f"✅ Deployed {stack} to {args.env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
