#!/usr/bin/env python3
"""Bootstrap the S3 bucket used by Terraform remote state.

Terraform cannot create the bucket that stores its own backend during the first
`terraform init`. Run this script once with AWS credentials in the environment;
it creates or hardens the bucket and writes a local `backend.hcl` file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_FILE = ROOT_DIR / "backend.hcl"


def run_aws(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["aws", *args]
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def aws_account_id() -> str:
    result = run_aws(["sts", "get-caller-identity", "--query", "Account", "--output", "text"])
    return result.stdout.strip()


def bucket_exists(bucket: str) -> bool:
    result = run_aws(["s3api", "head-bucket", "--bucket", bucket], check=False)
    return result.returncode == 0


def create_bucket(bucket: str, region: str) -> None:
    if bucket_exists(bucket):
        print(f"Bucket already exists: {bucket}")
        return

    print(f"Creating bucket: {bucket}")
    args = ["s3api", "create-bucket", "--bucket", bucket, "--region", region]

    if region != "us-east-1":
        args.extend(
            [
                "--create-bucket-configuration",
                f"LocationConstraint={region}",
            ]
        )

    run_aws(args)
    run_aws(["s3api", "wait", "bucket-exists", "--bucket", bucket])


def harden_bucket(bucket: str) -> None:
    print("Enabling bucket versioning")
    run_aws(
        [
            "s3api",
            "put-bucket-versioning",
            "--bucket",
            bucket,
            "--versioning-configuration",
            "Status=Enabled",
        ]
    )

    print("Enabling default server-side encryption")
    run_aws(
        [
            "s3api",
            "put-bucket-encryption",
            "--bucket",
            bucket,
            "--server-side-encryption-configuration",
            (
                '{"Rules":[{"ApplyServerSideEncryptionByDefault":'
                '{"SSEAlgorithm":"AES256"}}]}'
            ),
        ]
    )

    print("Blocking public access")
    run_aws(
        [
            "s3api",
            "put-public-access-block",
            "--bucket",
            bucket,
            "--public-access-block-configuration",
            (
                "BlockPublicAcls=true,IgnorePublicAcls=true,"
                "BlockPublicPolicy=true,RestrictPublicBuckets=true"
            ),
        ]
    )


def write_backend_file(bucket: str, region: str, key: str) -> None:
    content = "\n".join(
        [
            f'bucket       = "{bucket}"',
            f'key          = "{key}"',
            f'region       = "{region}"',
            "encrypt      = true",
            "use_lockfile = true",
            "",
        ]
    )
    BACKEND_FILE.write_text(content, encoding="utf-8")
    print(f"Wrote {BACKEND_FILE}")


def parse_args() -> argparse.Namespace:
    default_region = (
        os.getenv("TF_STATE_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    default_project = os.getenv("TF_STATE_PROJECT", "fiap-mlet")
    default_key = os.getenv("TF_STATE_KEY", f"{default_project}/terraform.tfstate")

    parser = argparse.ArgumentParser(
        description="Create the S3 bucket used by Terraform backend state."
    )
    parser.add_argument("--region", default=default_region)
    parser.add_argument("--project", default=default_project)
    parser.add_argument("--key", default=default_key)
    parser.add_argument("--bucket", default=os.getenv("TF_STATE_BUCKET"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        account_id = aws_account_id()
        bucket = args.bucket or f"{args.project}-terraform-state-{account_id}-{args.region}"

        create_bucket(bucket, args.region)
        harden_bucket(bucket)
        write_backend_file(bucket, args.region, args.key)

        print("\nNext steps:")
        print("  terraform init -backend-config=backend.hcl")
        print("  terraform plan")
        return 0
    except FileNotFoundError:
        print(
            "AWS CLI was not found. Install it or run this script in an environment "
            "where the `aws` command is available.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or exc.stdout.strip(), file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
