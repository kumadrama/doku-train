from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


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
    assert "AWS_ACCESS_KEY_ID=" not in readme


def test_readme_relative_links_resolve_to_repository_files() -> None:
    readme = Path("README.md").read_text()
    targets = re.findall(r"(?<!!)\[[^]]+]\(([^)]+)\)", readme)
    assert targets
    for target in targets:
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#"):
            continue
        path = Path(unquote(parsed.path))
        assert path.is_file(), f"README link does not resolve: {target}"
