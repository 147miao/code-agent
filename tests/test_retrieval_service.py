"""检索服务测试：repo CRUD、search 过滤/校验、symbol、overview、impact。"""
from __future__ import annotations

from pathlib import Path

import pytest

from codeagent.index.indexer import CodeIndexer
from codeagent.index.store import CodeStore
from codeagent.retrieval.service import (
    CodeRepoNotFound,
    CodeService,
    CodeValidationError,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _service(tmp_path: Path) -> tuple[CodeService, str]:
    root = tmp_path / "repo"
    _write(root / "pkg" / "a.py", "from .b import ToolExecutor\n\ndef make():\n    return ToolExecutor()\n")
    module_source = (
        '"""Demo module."""\n\n\n'
        "class ToolExecutor:\n"
        '    """Run tools."""\n\n'
        "    def run(self, name):\n"
        "        return name\n"
    )
    _write(root / "pkg" / "b.py", module_source)
    db = tmp_path / "code.sqlite"
    report = CodeIndexer(db, [tmp_path]).index(str(root), "demo")
    return CodeService(CodeStore(db)), report["repo_id"]


def test_repo_crud(tmp_path: Path) -> None:
    service, repo_id = _service(tmp_path)
    repos = service.list_repos()
    assert len(repos) == 1
    assert repos[0]["repo_id"] == repo_id and repos[0]["name"] == "demo"
    assert service.get_repo(repo_id)["status"] == "ready"
    assert service.get_repo("missing") is None
    assert service.delete_repo(repo_id) is True
    assert service.delete_repo(repo_id) is False
    assert service.list_repos() == []


def test_search_code_and_filters(tmp_path: Path) -> None:
    service, repo_id = _service(tmp_path)
    hits = service.search_code("ToolExecutor", k=8, repo_id=repo_id)
    assert hits
    assert all(hit["repo_id"] == repo_id for hit in hits)
    assert all(
        {"repo_id", "chunk_id", "path", "qualname", "kind", "start_line", "end_line", "score", "text"}
        == set(hit)
        for hit in hits
    )
    assert hits[0]["score"] > 0

    method_hits = service.search_code("ToolExecutor", k=8, repo_id=repo_id, kind="method")
    assert [hit["qualname"] for hit in method_hits] == ["ToolExecutor.run"]
    assert service.search_code("ToolExecutor", k=8, repo_id=repo_id, path_prefix="other") == []

    with pytest.raises(CodeValidationError):
        service.search_code("   ")
    with pytest.raises(CodeRepoNotFound):
        service.search_code("ToolExecutor", repo_id="repo_missing")
    assert len(service.search_code("ToolExecutor", k=0, repo_id=repo_id)) == 1


def test_get_symbol_and_overview(tmp_path: Path) -> None:
    service, repo_id = _service(tmp_path)
    symbol = service.get_symbol(repo_id, "ToolExecutor.run")
    assert symbol is not None
    assert symbol["path"] == "pkg/b.py"
    assert symbol["kind"] == "method"
    assert symbol["code"].lstrip().startswith("def run(")
    assert symbol["docstring"] == ""
    assert service.get_symbol(repo_id, "ToolExecutor.run", path="pkg/other.py") is None
    assert service.get_symbol(repo_id, "missing") is None

    overview = service.overview(repo_id)
    assert overview["repo_id"] == repo_id
    assert overview["name"] == "demo"
    assert overview["file_count"] == 2
    assert overview["symbol_count"] > 0
    assert {"class", "function", "method", "module"} <= set(overview["kinds"])
    assert overview["top_paths"] and overview["top_paths"][0]["path"].endswith(".py")

    with pytest.raises(CodeValidationError):
        service.get_symbol("", "x")
    with pytest.raises(CodeValidationError):
        service.overview("")


def test_impact_reverse_imports(tmp_path: Path) -> None:
    service, repo_id = _service(tmp_path)
    impact = service.impact(repo_id, "pkg/b.py")
    assert impact["repo_id"] == repo_id
    assert impact["path"] == "pkg/b.py"
    assert impact["imports"] == ["pkg/a.py"]
    assert impact["heuristic"] is True
    assert "反向 import" in impact["note"]

    assert service.reverse_imports(repo_id, "pkg/b.py") == ["pkg/a.py"]
    assert service.impact(repo_id, "pkg/missing.py")["imports"] == []

    with pytest.raises(CodeRepoNotFound):
        service.impact("repo_missing", "pkg/b.py")
    with pytest.raises(CodeValidationError):
        service.impact(repo_id, "")
