#!/usr/bin/env python3
"""Rollback helper for model packages.

Capabilities:
  - List model packages in S3 by environment (newest first)
  - Select a previous package and download it locally
  - Optionally update an S3 alias pointer (current.txt) to mark the active package

Notes:
  - Expects the S3 bucket name via --bucket or env S3_MODEL_ARTIFACTS_BUCKET
  - Package layout per packaging script: s3://<bucket>/model-packages/<env>/model_package-<ts>.zip
  - Alias pointer key: s3://<bucket>/model-packages/<env>/current.txt (contains the package key)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PackageInfo:
    key: str
    last_modified: str
    size: int


def _require_boto3():
    try:
        import boto3  # noqa: F401
        from botocore.exceptions import ClientError  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "boto3 is required for rollback operations. Install it in your Poetry env or run: \n"
            "poetry run pip install boto3\n"
            f"Original import error: {e}"
        )


def _get_s3_client():
    _require_boto3()
    import boto3

    return boto3.client("s3")


def list_packages(bucket: str, env: str) -> List[PackageInfo]:
    s3 = _get_s3_client()
    prefix = f"model-packages/{env}/"
    paginator = s3.get_paginator("list_objects_v2")
    items: List[PackageInfo] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith(".zip"):
                items.append(
                    PackageInfo(
                        key=key,
                        last_modified=obj["LastModified"].isoformat(),
                        size=obj["Size"],
                    )
                )
    # Newest first by LastModified
    items.sort(key=lambda x: x.last_modified, reverse=True)
    return items


def download_package(bucket: str, key: str, dest_dir: str) -> str:
    s3 = _get_s3_client()
    os.makedirs(dest_dir, exist_ok=True)
    filename = key.rsplit("/", 1)[-1]
    dest_path = os.path.join(dest_dir, filename)
    s3.download_file(bucket, key, dest_path)
    print(f"📥 Downloaded: s3://{bucket}/{key} -> {dest_path}")
    return dest_path


def set_current_alias(bucket: str, env: str, key: str) -> None:
    s3 = _get_s3_client()
    alias_key = f"model-packages/{env}/current.txt"
    s3.put_object(
        Bucket=bucket,
        Key=alias_key,
        Body=key.encode("utf-8"),
        ContentType="text/plain",
    )
    print(f"🔗 Set alias: s3://{bucket}/{alias_key} -> {key}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Model rollback helper")
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Target environment")
    parser.add_argument("--bucket", default=os.getenv("S3_MODEL_ARTIFACTS_BUCKET"), help="S3 bucket name")
    parser.add_argument("--list", action="store_true", help="List available packages and exit")
    parser.add_argument("--previous", action="store_true", help="Automatically select previous package (index 1)")
    parser.add_argument("--key", default=None, help="Explicit package key to operate on")
    parser.add_argument("--download", action="store_true", help="Download the selected package to artifacts/<env>/")
    parser.add_argument("--set-alias", action="store_true", help="Set current.txt alias to the selected package key")
    args = parser.parse_args(argv)

    if not args.bucket:
        raise SystemExit("Missing S3 bucket. Provide --bucket or set S3_MODEL_ARTIFACTS_BUCKET.")

    pkgs = list_packages(args.bucket, args.env)
    if args.list:
        if not pkgs:
            print("No packages found.")
            return 0
        print(f"Found {len(pkgs)} package(s) for {args.env} (newest first):")
        for i, p in enumerate(pkgs):
            print(f"[{i}] {p.key}  {p.size}B  {p.last_modified}")
        return 0

    # Resolve selection
    selected_key: Optional[str] = args.key
    if not selected_key:
        if not pkgs:
            raise SystemExit("No packages to select.")
        idx = 1 if args.previous and len(pkgs) > 1 else 0
        selected_key = pkgs[idx].key
        print(f"Selected package: {selected_key}")

    # Download
    if args.download:
        dest = os.path.join("artifacts", args.env)
        download_package(args.bucket, selected_key, dest)

    # Update alias
    if args.set_alias:
        set_current_alias(args.bucket, args.env, selected_key)

    if not args.download and not args.set_alias:
        print("No action specified (use --download and/or --set-alias).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
