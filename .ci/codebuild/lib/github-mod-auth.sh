#!/usr/bin/env bash
# Configure git so go can fetch private modules under github.com/kumadrama/*.
# CodeBuild's git-credential-helper only vends credentials cleanly for the
# project's primary source repo; sibling repos (e.g. doku-lib) get HTTP 400.
# We pull a token for the primary source, then apply it host-wide.
set -euo pipefail

export GOPRIVATE="${GOPRIVATE:-github.com/kumadrama/*}"
export GONOSUMDB="${GONOSUMDB:-github.com/kumadrama/*}"

HELPER="/codebuild/readonly/bin/git-credential-helper"
if [[ ! -x "${HELPER}" ]]; then
  echo "github-mod-auth: missing ${HELPER} (enable env.git-credential-helper: yes)" >&2
  exit 1
fi

OWNER_REPO=$(printf '%s' "${CODEBUILD_SOURCE_REPO_URL:-}" | sed -E 's#.*github.com[:/]([^/]+/[^/.]+)(\.git)?$#\1#')
if [[ -n "${OWNER_REPO}" && "${OWNER_REPO}" != "${CODEBUILD_SOURCE_REPO_URL:-}" ]]; then
  CRED_OUT=$(printf 'protocol=https\nhost=github.com\npath=%s\n\n' "${OWNER_REPO}" | "${HELPER}" get)
else
  CRED_OUT=$(printf 'protocol=https\nhost=github.com\n\n' | "${HELPER}" get)
fi

GH_TOKEN=$(printf '%s\n' "${CRED_OUT}" | awk -F= '/^password=/{print substr($0,10); exit}')
if [[ -z "${GH_TOKEN}" ]]; then
  echo "github-mod-auth: CodeConnections token unavailable" >&2
  exit 1
fi

echo "github-mod-auth: configured (token_len=${#GH_TOKEN})"

git config --global --unset-all credential.helper || true
git config --global credential.helper ""
printf 'machine github.com\nlogin x-access-token\npassword %s\n' "${GH_TOKEN}" >"${HOME}/.netrc"
chmod 600 "${HOME}/.netrc"
AUTH_B64=$(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 | tr -d '\n')
git config --global http.https://github.com/.extraheader "AUTHORIZATION: basic ${AUTH_B64}"
