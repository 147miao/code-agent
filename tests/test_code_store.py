"""CodeStore 测试：CRUD、增量行、双 FTS5、过滤、符号、反向 import、概览、reindex。"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from codeagent.index.chunker import build_chunks
from codeagent.index.parser import ModuleInfo, parse_source
from codeagent.index.store import CodeStore

SOURCE = '''"""Demo module."""

import os


class ToolExecutor:
    """Run tools."""

    def run(self, name):
        return os.path.basename(name)


def helper():
    return ToolExecutor()
'''


def _seed(store: CodeStore, tmp_path: Path, path: str = "pkg/mod.py") -> tuple[str, ModuleInfo]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    repo_id = store.add_repo(str(repo_root), "demo")["repo_id"]
    info = parse_source(path, SOURCE)
    assert info.error is None
    store.upsert_file(repo_id, path, "hash1", len(SOURCE.encode()), 1.0)
    store.insert_symbols(repo_id, path, [asdict(symbol) for symbol in info.symbols])
    store.insert_imports(repo_id, path, [asdict(item) for item in info.imports])
    store.insert_references(repo_id, path, [asdict(item) for item in info.references])
    store.insert_chunks(repo_id, build_chunks(repo_id, path, info.symbols))
    return repo_id, info


@pytest.fixture
def store(tmp_path: Path) -> CodeStore:
    return CodeStore(tmp_path / "code_index.sqlite")


def test_repo_crud_and_idempotent_add(store: CodeStore, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    repo = store.add_repo(str(root), "demo")
    assert repo["repo_id"].startswith("repo_")
    assert repo["file_count"] == 0 and repo["symbol_count"] == 0
    assert repo["status"] == "pending"

    again = store.add_repo(str(root), "renamed")
    assert again["repo_id"] == repo["repo_id"]
    assert again["name"] == "renamed"
    assert len(store.list_repos()) == 1

    assert store.update_repo_stats(repo["repo_id"], file_count=2, symbol_count=3, status="ready")
    saved = store.get_repo(repo["repo_id"])
    assert saved is not None
    assert (saved["file_count"], saved["symbol_count"], saved["status"]) == (2, 3, "ready")

    assert store.delete_repo(repo["repo_id"]) is True
    assert store.get_repo(repo["repo_id"]) is None
    assert store.list_repos() == []
    assert store.delete_repo(repo["repo_id"]) is False


def test_file_rows_and_delete(store: CodeStore, tmp_path: Path) -> None:
    repo_id, _ = _seed(store, tmp_path)
    path = "pkg/mod.py"

    assert store.get_file_hash(repo_id, path) == "hash1"
    assert store.get_file_hash(repo_id, "missing.py") is None
    assert [item["path"] for item in store.list_files(repo_id)] == [path]

    overview = store.overview(repo_id)
    assert overview["file_count"] == 1
    assert overview["symbol_count"] == 4  # module + class + method + function
    assert overview["chunk_count"] == 3

    store.delete_file_rows(repo_id, path)
    assert store.get_file_hash(repo_id, path) is None
    cleared = store.overview(repo_id)
    assert cleared["file_count"] == 0
    assert cleared["symbol_count"] == 0
    assert cleared["chunk_count"] == 0
    assert store.search("ToolExecutor") == []


def test_search_rrf_and_parameterized_filters(store: CodeStore, tmp_path: Path) -> None:
    repo_id, _ = _seed(store, tmp_path)
    hits = store.search("ToolExecutor", k=8)
    assert hits
    assert any(hit["qualname"] == "ToolExecutor" for hit in hits)
    assert all(hit["repo_id"] == repo_id for hit in hits)
    assert all(
        {"chunk_id", "path", "qualname", "kind", "start_line", "end_line", "text", "score"} <= set(hit)
        for hit in hits
    )
    assert hits[0]["score"] > 0

    method_hits = store.search("ToolExecutor", kind="method")
    assert [hit["qualname"] for hit in method_hits] == ["ToolExecutor.run"]
    assert store.search("ToolExecutor", kind="module") == []

    assert store.search("ToolExecutor", path_prefix="pkg")
    assert store.search("ToolExecutor", path_prefix="pkg/mod.py")
    assert store.search("ToolExecutor", path_prefix="other") == []

    assert store.search("ToolExecutor", repo_id="repo_other") == []
    assert store.search("ToolExecutor", repo_id=repo_id + "' OR 1=1 --") == []
    assert store.search("") == []
    assert store.search("%' --; DROP TABLE chunks; --") == []


def test_get_symbol_exact(store: CodeStore, tmp_path: Path) -> None:
    repo_id, _ = _seed(store, tmp_path)
    symbol = store.get_symbol(repo_id, "ToolExecutor.run")
    assert symbol is not None
    assert symbol["path"] == "pkg/mod.py"
    assert symbol["kind"] == "method"
    assert symbol["code"].lstrip().startswith("def run(")
    assert symbol["start_line"] <= symbol["end_line"]
    assert store.get_symbol(repo_id, "ToolExecutor.run", path="pkg/mod.py") == symbol
    assert store.get_symbol(repo_id, "ToolExecutor.run", path="other.py") is None
    assert store.get_symbol(repo_id, "missing") is None
    assert store.get_symbol("repo_other", "ToolExecutor.run") is None


def test_reverse_imports_heuristic(store: CodeStore, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    repo_id = store.add_repo(str(root), "demo")["repo_id"]
    store.insert_imports(repo_id, "pkg/a.py", [{"module": "pkg.b", "level": 0, "lineno": 1}])
    store.insert_imports(repo_id, "pkg/c.py", [{"module": "b", "level": 1, "lineno": 1}])
    store.insert_imports(repo_id, "pkg/d.py", [{"module": "b_extra", "level": 0, "lineno": 1}])
    store.insert_imports(repo_id, "pkg/b.py", [{"module": "pkg.b", "level": 0, "lineno": 1}])

    assert store.reverse_imports(repo_id, "pkg/b.py") == ["pkg/a.py", "pkg/c.py"]
    assert store.reverse_imports(repo_id, "pkg/missing.py") == []

    store.insert_imports(repo_id, "pkg/e.py", [{"module": "pkg", "level": 0, "lineno": 1}])
    assert store.reverse_imports(repo_id, "pkg/__init__.py") == ["pkg/e.py"]


def test_overview_and_reindex(store: CodeStore, tmp_path: Path) -> None:
    repo_id, _ = _seed(store, tmp_path)
    overview = store.overview(repo_id)
    assert overview["file_count"] == 1
    assert overview["module_count"] == 1
    assert overview["class_count"] == 1
    assert overview["function_count"] == 1
    assert overview["method_count"] == 1
    assert overview["symbol_count"] == 4
    assert overview["kinds"]["class"] == 1 and overview["kinds"]["function"] == 1
    assert overview["top_paths"] == [{"path": "pkg/mod.py", "symbols": 4}]
    assert overview["import_count"] == 1
    assert overview["reference_count"] == 3  # os / name / ToolExecutor
    assert overview["chunk_count"] == 3

    assert store.reindex_repo(repo_id) is True
    cleared = store.overview(repo_id)
    assert cleared["file_count"] == 0 and cleared["symbol_count"] == 0
    assert cleared["status"] == "pending"
    assert store.reindex_repo("repo_other") is False
    assert store.overview("repo_other") == {}


def test_delete_repo_cascades_chunks_and_fts(store: CodeStore, tmp_path: Path) -> None:
    repo_id, _ = _seed(store, tmp_path)
    assert store.search("ToolExecutor")
    assert store.delete_repo(repo_id) is True
    assert store.search("ToolExecutor") == []


def test_sqlite_pragmas_and_fts_tables(store: CodeStore) -> None:
    conn = sqlite3.connect(store.db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert store.busy_timeout_ms == 3000
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')")
        }
        assert {"repos", "files", "symbols", "imports", "references", "chunks"} <= names
        assert {"chunks_fts", "chunks_fts_trigram"} <= names
        triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
        assert {"chunks_ai", "chunks_ad", "chunks_au"} <= triggers
    finally:
        conn.close()
