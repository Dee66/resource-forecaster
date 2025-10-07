#!/usr/bin/env python3
"""Package model artifacts and inference code for deployment.

Features:
- Collects model files and inference code into a versioned zip
- Generates metadata.json with build info
- Computes SHA256 checksum for supply-chain integrity
- Optional S3 upload if --s3-bucket is provided (requires boto3 and AWS creds)

Usage examples:
  poetry run python scripts/package_forecaster_artifacts.py --env dev
  poetry run python scripts/package_forecaster_artifacts.py --env staging --s3-bucket my-artifacts-bucket
  poetry run python scripts/package_forecaster_artifacts.py --env prod --include src/forecaster/inference --include models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _get_git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return os.environ.get("GITHUB_SHA")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path, include_hidden: bool = False) -> None:
    if not src.exists():
        return
    for root, dirs, files in os.walk(src):
        rpath = Path(root)
        # Skip hidden if requested
        if not include_hidden and any(part.startswith(".") for part in rpath.relative_to(src).parts):
            continue
        rel = rpath.relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            if not include_hidden and name.startswith("."):
                continue
            shutil.copy2(rpath / name, dst / rel / name)


def package(env: str, includes: Iterable[Path], output_dir: Path, s3_bucket: str | None) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    work_dir = ROOT / ".build" / "package" / env / ts
    out_dir = output_dir / env
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy includes
    for p in includes:
        src = (ROOT / p).resolve()
        if src.is_file():
            dst = work_dir / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif src.is_dir():
            _copy_tree(src, work_dir / p)

    # Add environment config if present
    cfg = ROOT / "config" / f"{env}.yml"
    if cfg.exists():
        dst = work_dir / "config" / cfg.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, dst)

    # Ensure metadata
    metadata = {
        "environment": env,
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _get_git_sha(),
        "includes": [str(p) for p in includes],
        "tool": "package_forecaster_artifacts.py",
        "version": 1,
    }
    (work_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Create zip
    zip_path = out_dir / f"model_package-{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(work_dir):
            for name in files:
                fpath = Path(root) / name
                arc = fpath.relative_to(work_dir)
                zf.write(fpath, arcname=str(arc))

    # Write checksum
    checksum = _sha256_of_file(zip_path)
    (zip_path.with_suffix(".sha256")).write_text(checksum + "\n", encoding="utf-8")

    # Optional S3 upload
    if s3_bucket:
        try:
            import boto3  # type: ignore

            s3 = boto3.client("s3")
            key_prefix = f"model-packages/{env}/"
            s3.upload_file(str(zip_path), s3_bucket, key_prefix + zip_path.name, ExtraArgs={"ContentType": "application/zip"})
            # Upload checksum and metadata
            s3.upload_file(str(zip_path.with_suffix(".sha256")), s3_bucket, key_prefix + zip_path.with_suffix(".sha256").name, ExtraArgs={"ContentType": "text/plain"})
            s3.upload_file(str(work_dir / "metadata.json"), s3_bucket, key_prefix + "metadata-" + ts + ".json", ExtraArgs={"ContentType": "application/json"})
            print(f"✅ Uploaded artifacts to s3://{s3_bucket}/{key_prefix}")
        except Exception as e:
            print(f"⚠️  S3 upload skipped or failed: {e}")

    print(f"📦 Package created: {zip_path}")
    print(f"🔐 SHA256: {checksum}")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package forecaster artifacts")
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Target environment")
    parser.add_argument("--include", action="append", default=[], help="Relative path to include (file or dir). Can be used multiple times.")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory for packaged zip")
    parser.add_argument("--s3-bucket", default=None, help="Optional S3 bucket to upload the package")
    args = parser.parse_args(argv)

    # Defaults if no includes are provided
    includes = [
        Path("src/forecaster/inference"),
        Path("src/forecaster/config.py"),
        Path("src/forecaster/exceptions.py"),
        Path("models"),  # optional
        Path("requirements.txt"),
        Path("pyproject.toml"),
    ] if not args.include else [Path(p) for p in args.include]

    try:
        package(
            env=args.env,
            includes=includes,
            output_dir=(ROOT / args.output_dir),
            s3_bucket=args.s3_bucket,
        )
        return 0
    except Exception as e:
        print(f"❌ Packaging failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
