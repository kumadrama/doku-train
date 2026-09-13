# S3-first Developer README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a Chinese, S3-first root README that lets an algorithm developer validate input and launch a real CPU rerank training run with copyable commands.

**Architecture:** Keep `README.md` as an onboarding/runbook layer over the existing architecture and Spec documents. Reuse the single Typer composition root, adding only Python module execution; static README contract tests keep commands, four outputs, links, security boundaries, and the pending production benchmark honest.

**Tech Stack:** Markdown, Python 3.12, Typer, pytest, uv, Make

---

### Task 1: Make the documented module command executable

**Files:**

- Modify: `recommend/train/run.py`
- Modify: `recommend/train/tests/unit/test_cli.py`

- [x] **Step 1: Write the failing module-entry test**

Add a subprocess test that runs the same interpreter and requires help output:

```python
def test_python_module_entrypoint_displays_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "recommend.train.run", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "rerank" in result.stdout
```

- [x] **Step 2: Run the test and observe RED**

Run: `.venv/bin/python -m pytest recommend/train/tests/unit/test_cli.py::test_python_module_entrypoint_displays_cli_help -q`

Expected: FAIL because importing `recommend.train.run` exits without invoking `app`.

- [x] **Step 3: Add the only CLI module entry**

Append to `recommend/train/run.py`:

```python
if __name__ == "__main__":
    app()
```

- [x] **Step 4: Run the focused CLI tests**

Run: `.venv/bin/python -m pytest recommend/train/tests/unit/test_cli.py -q`

Expected: all CLI tests PASS.

- [x] **Step 5: Commit**

```bash
git add recommend/train/run.py recommend/train/tests/unit/test_cli.py
git commit -m "fix: expose training module cli"
```

### Task 2: Add the S3-first developer README

**Files:**

- Create: `README.md`
- Create: `recommend/train/tests/unit/test_readme.py`

- [x] **Step 1: Write the failing README contract tests**

The test must require the S3-first commands, canonical outputs, documentation links, pending capacity wording,
and reject credential examples:

```python
def test_readme_documents_the_real_training_contract() -> None:
    readme = Path("README.md").read_text()
    for command in ("rerank validate", "rerank train", "rerank backtest"):
        assert command in readme
    for output in (
        "checkpoint.pt",
        "metrics.json",
        "fitted-feature-state/state.json",
        "lineage.json",
    ):
        assert output in readme
    assert "约 2800 万" in readme
    assert "尚未完成生产容量基准" in readme
    assert "AWS_SECRET_ACCESS_KEY=" not in readme
```

It must also parse every relative Markdown link and assert that the referenced repository file exists.

- [x] **Step 2: Run the test and observe RED**

Run: `.venv/bin/python -m pytest recommend/train/tests/unit/test_readme.py -q`

Expected: FAIL because root `README.md` does not exist.

- [x] **Step 3: Write `README.md` in the approved order**

Create these exact top-level sections:

```markdown
# doku-train
## 当前能力
## 首次真实训练
## 输入合同
## 训练配置
## 查看训练结果
## 常见失败
## Finder 目录对应
## 本地开发
## 文档索引
```

Use placeholder URIs such as `s3://<training-bucket>/rerank/...`; explain the standard AWS credential chain
without embedding keys. Show `uv sync --group dev`, then copyable `validate`, `train`, and `backtest` commands
through `uv run python -m recommend.train.run`. State that only four training outputs exist, AUC is warning-only,
ONNX/serving/GPU/multi-machine/cloud scheduling remain out of scope, and the 28-day/approximately 28M-row
production benchmark is pending rather than an SLA.

- [x] **Step 4: Run README and CLI tests**

Run: `.venv/bin/python -m pytest recommend/train/tests/unit/test_readme.py recommend/train/tests/unit/test_cli.py -q`

Expected: all tests PASS.

- [x] **Step 5: Commit**

```bash
git add README.md recommend/train/tests/unit/test_readme.py
git commit -m "docs: add s3-first developer readme"
```

### Task 3: Synchronize documentation status and verify

**Files:**

- Modify: `docs/architecture/developer-readme-design.md`
- Modify: `docs/architecture/system-overview.md`
- Modify: `docs/architecture/interfaces.md`
- Modify: `specs/002-s3-first-readme/tasks.md`

- [x] **Step 1: Replace stale implementation-status text**

Mark the README design implemented, and change architecture/interface status text from “尚未实现” to the
verified Spec 001 state while retaining the pending production capacity benchmark.

- [x] **Step 2: Mark this plan complete**

Change every task checkbox in this file from `[ ]` to `[x]` only after its command has passed.

- [x] **Step 3: Run full verification**

Run:

```bash
uv lock --check
make check
uv run python -m recommend.train.run --help
git diff --check
```

Expected: all commands exit 0; pytest has no failures; help contains `rerank`; Git working tree contains no
uncommitted files after the final commit.

- [x] **Step 4: Commit**

```bash
git add docs/architecture/developer-readme-design.md docs/architecture/system-overview.md docs/architecture/interfaces.md specs/002-s3-first-readme/tasks.md
git commit -m "docs: complete developer readme verification"
```
