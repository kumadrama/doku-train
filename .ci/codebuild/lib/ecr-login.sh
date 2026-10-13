#!/usr/bin/env bash
# docker login to the account's ECR registry in REGION.
set -euo pipefail

usage() {
  echo "usage: ecr-login.sh --region REGION" >&2
  exit 2
}

REGION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="${2:-}"
      shift 2
      ;;
    -h | --help) usage ;;
    *)
      echo "ecr-login: unknown arg: $1" >&2
      usage
      ;;
  esac
done

[[ -n "${REGION}" ]] || usage

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
echo "ecr-login: ${REGISTRY}"
