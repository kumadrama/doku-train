#!/usr/bin/env bash
# Allocate IMAGE_TAG: [dev-]YYYYMMDD.N-<7-char-commit>[-actor] on non-main.
# Write KEY=VALUE to --out. Commit is the latest that touched --service-dir
# or (when present) the repo-root go.mod.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: allocate-image-tags.sh \
  --ecr-repository NAME \
  --channel main|dev \
  --repo-root DIR \
  --region REGION \
  --out FILE \
  --service-dir REL \
  [--commit-sha SHA] \
  [--source-ref REF]
EOF
  exit 2
}

ECR_REPOSITORY=""
CHANNEL="main"
REPO_ROOT=""
REGION=""
OUT=""
SERVICE_DIR=""
COMMIT_SHA=""
SOURCE_REF=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ecr-repository) ECR_REPOSITORY="${2:-}"; shift 2 ;;
    --channel) CHANNEL="${2:-}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:-}"; shift 2 ;;
    --region) REGION="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --service-dir) SERVICE_DIR="${2:-}"; shift 2 ;;
    --commit-sha) COMMIT_SHA="${2:-}"; shift 2 ;;
    --source-ref) SOURCE_REF="${2:-}"; shift 2 ;;
    -h | --help) usage ;;
    *)
      echo "allocate-image-tags: unknown arg: $1" >&2
      usage
      ;;
  esac
done

[[ -n "${ECR_REPOSITORY}" && -n "${REPO_ROOT}" && -n "${REGION}" && -n "${OUT}" && -n "${SERVICE_DIR}" ]] || usage

case "${CHANNEL}" in
  main) TAG_PREFIX="" ;;
  *)
    CHANNEL="dev"
    TAG_PREFIX="dev-"
    ;;
esac

# Calendar day in UTC+8 (Asia/Shanghai), not UTC.
DATE_TAG=$(TZ=Asia/Shanghai date +%Y%m%d)
MAX=0
# Match today's date tags: [dev-]YYYYMMDD.N or [dev-]YYYYMMDD.N-<sha>
while IFS= read -r t; do
  [[ -z "${t}" ]] && continue
  case "${t}" in
    ${TAG_PREFIX}${DATE_TAG}.*)
      rest=${t#"${TAG_PREFIX}${DATE_TAG}."}
      N=${rest%%-*}
      if [[ "${N}" =~ ^[0-9]+$ ]] && ((N > MAX)); then
        MAX=${N}
      fi
      ;;
  esac
done < <(
  aws ecr describe-images \
    --repository-name "${ECR_REPOSITORY}" \
    --region "${REGION}" \
    --filter "tagStatus=TAGGED" \
    --query 'imageDetails[].imageTags[]' \
    --output text |
    tr '\t' '\n'
)

NEXT=$((MAX + 1))

# CodeBuild / GitHub often clone with --depth 1. On a shallow graft,
# `git log -1 -- <path>` can return HEAD even when that commit did not
# touch the path — so expand history before path-limited log.
ensure_full_history_for_path_log() {
  local root="$1"
  if [[ "$(git -C "${root}" rev-parse --is-shallow-repository 2>/dev/null || true)" != "true" ]]; then
    return 0
  fi
  echo "allocate-image-tags: shallow clone; fetching history for service-dir commit" >&2
  if git -C "${root}" fetch --unshallow 2>/dev/null; then
    return 0
  fi
  git -C "${root}" fetch --deepen 1000 2>/dev/null || true
}

# Fingerprint paths: service dir is required; root go.mod is included when
# present so a dependency bump rebuilds every Go service in the repo.
resolve_service_commit() {
  local root="$1"
  local service_dir="$2"
  local svc_sha
  svc_sha=$(git -C "${root}" log -1 --format=%H -- "${service_dir}")
  if [[ -z "${svc_sha}" ]]; then
    return 1
  fi
  local paths=("${service_dir}")
  if git -C "${root}" cat-file -e "HEAD:go.mod" 2>/dev/null; then
    paths+=("go.mod")
  fi
  git -C "${root}" log -1 --format=%H -- "${paths[@]}"
}

case "${COMMIT_SHA}" in
  "" | UNKNOWN | unknown)
    ensure_full_history_for_path_log "${REPO_ROOT}"
    COMMIT_SHA=$(resolve_service_commit "${REPO_ROOT}" "${SERVICE_DIR}")
    ;;
esac
COMMIT_SHA=$(printf '%s' "${COMMIT_SHA}" | tr -d '[:space:]')
[[ -n "${COMMIT_SHA}" ]] || {
  echo "allocate-image-tags: empty commit sha for service-dir=${SERVICE_DIR}" >&2
  exit 1
}
COMMIT_SHORT=$(printf '%s' "${COMMIT_SHA}" | cut -c1-7)
[[ -n "${COMMIT_SHORT}" ]] || {
  echo "allocate-image-tags: empty commit sha" >&2
  exit 1
}

IMAGE_TAG_BASE="${TAG_PREFIX}${DATE_TAG}.${NEXT}-${COMMIT_SHORT}"
IMAGE_TAG="${IMAGE_TAG_BASE}"

# Actor is injected by `dkd ci build` (DOKU_IMAGE_ACTOR). Never resolve gh/user here:
# CodeBuild's gh is the GitHub connector, not the operator.
# On non-main, IMAGE_TAG is the actor-suffixed tag (what docker-build / first push uses).
# IMAGE_TAG_ACTOR is the unsuffixed alias so skip-build and older pickers still match.
IMAGE_TAG_ACTOR=""
if [[ "${CHANNEL}" != "main" ]]; then
  ACTOR="$(printf '%s' "${DOKU_IMAGE_ACTOR:-}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"
  ACTOR="${ACTOR#-}"
  ACTOR="${ACTOR%-}"
  if ((${#ACTOR} > 32)); then
    ACTOR="${ACTOR:0:32}"
    ACTOR="${ACTOR%-}"
  fi
  if [[ "${ACTOR}" =~ ^[a-z][a-z0-9-]{0,31}$ ]]; then
    IMAGE_TAG="${IMAGE_TAG_BASE}-${ACTOR}"
    IMAGE_TAG_ACTOR="${IMAGE_TAG_BASE}"
  fi
fi

echo "allocate-image-tags: channel=${CHANNEL} source_ref=${SOURCE_REF:-} service_dir=${SERVICE_DIR} commit=${COMMIT_SHORT} image_tag=${IMAGE_TAG} alias_tag=${IMAGE_TAG_ACTOR:-}"

umask 077
{
  printf 'BUILD_CHANNEL=%s\n' "${CHANNEL}"
  printf 'IMAGE_TAG=%s\n' "${IMAGE_TAG}"
  if [[ -n "${IMAGE_TAG_ACTOR}" ]]; then
    printf 'IMAGE_TAG_ACTOR=%s\n' "${IMAGE_TAG_ACTOR}"
  fi
} >"${OUT}"
