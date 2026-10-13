#!/usr/bin/env bash
# Tag PRIMARY as SECONDARY (if given), push all listed images, write URIs to --out.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: push-ecr-images.sh --out FILE PRIMARY_URI [SECONDARY_URI ...]
  PRIMARY must already exist locally. Each additional URI is created via
  docker tag from PRIMARY, then all URIs are pushed.
EOF
  exit 2
}

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="${2:-}"
      shift 2
      ;;
    -h | --help) usage ;;
    --)
      shift
      break
      ;;
    -*)
      echo "push-ecr-images: unknown arg: $1" >&2
      usage
      ;;
    *)
      break
      ;;
  esac
done

[[ -n "${OUT}" ]] || usage
[[ $# -ge 1 ]] || usage

PRIMARY="$1"
shift
URIS=("${PRIMARY}")

for extra in "$@"; do
  docker tag "${PRIMARY}" "${extra}"
  URIS+=("${extra}")
done

for uri in "${URIS[@]}"; do
  docker push "${uri}"
  echo "push-ecr-images: pushed ${uri}"
done

umask 077
{
  for uri in "${URIS[@]}"; do
    printf '%s\n' "${uri}"
  done
} >"${OUT}"
