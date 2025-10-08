#!/usr/bin/env python3
"""Budget guardrail checker: Compare predicted cost to budget and optionally block deploys.

This script queries a local metadata JSON (or can call Cost Explorer) to read predicted
costs and compares them to a configured budget. If predicted cost > budget, it exits
non-zero (useful to block CDK deploys in pipeline).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


def load_prediction(path: str) -> dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted-json", default="dashboards/predicted_cost.json")
    parser.add_argument("--budget", type=float, default=1000.0, help="Budget threshold in USD")
    parser.add_argument("--env", default="prod")
    parser.add_argument("--block-exit", action="store_true", help="Exit non-zero when over budget")
    args = parser.parse_args(argv)

    if not os.path.exists(args.predicted_json):
        print(f"⚠️  Predicted cost file not found: {args.predicted_json}")
        return 2

    data = load_prediction(args.predicted_json)
    # Expect structure: {"envs": {"prod": {"predicted_monthly": 1234.12}}}
    envs = data.get("envs", {})
    env_data = envs.get(args.env, {})
    predicted = env_data.get("predicted_monthly")
    if predicted is None:
        print(f"⚠️  No predicted value for env {args.env} in {args.predicted_json}")
        return 2

    print(f"🔎 Predicted monthly cost for {args.env}: ${predicted:.2f}")
    # Determine budget: priority order
    # 1. Environment variable BUDGET_<ENV> (e.g., BUDGET_PROD)
    # 2. Environment variable BUDGET
    # 3. CLI --budget
    env_key = f"BUDGET_{args.env.upper()}"
    env_budget = os.environ.get(env_key)
    global_budget = os.environ.get("BUDGET")

    if env_budget is not None:
        try:
            budget = float(env_budget)
        except ValueError:
            print(f"⚠️  Invalid environment budget value for {env_key}: {env_budget}")
            return 2
    elif global_budget is not None:
        try:
            budget = float(global_budget)
        except ValueError:
            print(f"⚠️  Invalid environment budget value for BUDGET: {global_budget}")
            return 2
    else:
        budget = args.budget

    print(f"🎯 Budget threshold: ${budget:.2f}")
    if predicted > budget:
        print(f"⛔ Predicted cost exceeds budget by ${predicted-budget:.2f}")
        if args.block_exit:
            return 3
    else:
        print("✅ Predicted cost is within budget")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
