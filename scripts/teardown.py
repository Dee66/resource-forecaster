#!/usr/bin/env python3
"""Teardown script to destroy Resource Forecaster stacks safely."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Destroy CDK stacks with confirmation")
    p.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Target environment")
    p.add_argument("--yes", action="store_true", help="Auto-confirm destroy")
    args = p.parse_args(argv)

    suffix = args.env.capitalize()
    stack = f"ResourceForecaster-{suffix}"

    if not args.yes:
        print(f"⚠️  You are about to destroy stack: {stack}")
        return 2

    infra_dir = ROOT / "infra"
    code = run(["poetry", "run", "cdk", "destroy", stack, "--force"], cwd=infra_dir)
    if code == 0:
        print(f"🧹 Destroyed stack: {stack}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
