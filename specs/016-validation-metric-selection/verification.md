# verification

Validation evidence will be recorded after running checks. Experiment YAML and cloud results remain local under runs/.

2026-10-22: `UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv make check` passed: lock check, format, lint, mypy and 225 tests passed, 3 skipped. New synthetic training tests verify six full epochs despite intervening degradation, min/max metric selection, undefined metric handling and exact restoration of selected model weights. Unknown selection metrics are rejected. `git diff --check` passed.
