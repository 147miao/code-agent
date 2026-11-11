"""索引器测试：report、增量改/删、幂等、reindex、allowed_roots、限制跳过、CLI。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import codeagent.index.indexer as indexer_module
from codeagent.index import scanner as scanner_module
from codeagent.index.chunker import MAX_CHUNK_CHARS, MAX_CHUNK_LINES, build_chunks
from codeagent.index.indexer import CodeIndexer
from codeagent.index.parser import parse_source
from codeagent.index.scanner import REASON_TOO_LARGE, REASON_TOO_MANY_LINES, RootNotAllowedError
from codeagent.index.store import CodeStore

ROOT = Path(__file__).resolve().parents[1]
REPORT_KEYS = {
    "repo_id",
    "files",
    "symbols",
    "imports",
    "references",
    "chunks",
    "skipped",
    "errors",
    "ms",
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root / "pkg" / "__init__.py", "")
    _write(root / "pkg" / "a.py", "from .b import ToolExecutor\n\ndef make():\n    return ToolExecutor()\n")
    _write(root / "pkg" / "b.py", "class ToolExecutor:\n    def run(self):\n        return 1\n")
    _write(root / "bad.py", "def broken(:\n")
    _write(root / ".venv" / "hidden.py", "HIDDEN = 1\n")
    _write(root / "data" / "hidden.py", "HIDDEN = 1\n")
    return root


def test_index_report_and_idempotent(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    indexer = CodeIndexer(tmp_path / "code.sqlite", [tmp_path])

    first = indexer.index(str(root), "demo")
    assert set(first) == REPORT_KEYS
    assert first["repo_id"].startswith("repo_")
    assert first["files"] == 3  # __init__.py + a.py + b.py；bad.py 语法错误
    assert first["symbols"] == 6  # 3 个 module + class + method + function
    assert first["imports"] == 1
    assert first["references"] == 1  # a.py 中 ToolExecutor
    assert first["chunks"] == 3  # make / ToolExecutor / ToolExecutor.run；空 __init__ 无 chunk
    assert first["skipped"] == []
    assert [item["path"] for item in first["errors"]] == ["bad.py"]
    assert first["ms"] >= 0

    store = CodeStore(tmp_path / "code.sqlite")
    repo = store.get_repo(first["repo_id"])
    assert repo is not None and repo["status"] == "ready"
    hits = store.search("ToolExecutor", k=100)
    assert hits and any(hit["path"] == "pkg/b.py" for hit in hits)
    assert all(hit["path"] in {"pkg/a.py", "pkg/b.py"} for hit in hits)

    second = indexer.index(str(root), "demo")
    assert second["repo_id"] == first["repo_id"]
    for key in ("files", "symbols", "imports", "references", "chunks"):
        assert second[key] == first[key]
    assert len(store.search("ToolExecutor", k=100)) == len(hits)


def test_unchanged_files_are_not_reparsed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "A = 1\n")
    _write(root / "b.py", "B = 2\n")
    parsed: list[str | None] = []
    real_parse = indexer_module.parse_file

    def spy(path, *, path_label=None):
        parsed.append(path_label)
        return real_parse(path, path_label=path_label)

    monkeypatch.setattr(indexer_module, "parse_file", spy)
    indexer = CodeIndexer(tmp_path / "code.sqlite", [tmp_path])
    indexer.index(str(root), "demo")
    assert sorted(parsed) == ["a.py", "b.py"]

    indexer.index(str(root), "demo")
    assert sorted(parsed) == ["a.py", "b.py"]


def test_incremental_change_and_delete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def old_func():\n    return 1\n")
    _write(root / "b.py", "def keep_func():\n    return 2\n")
    indexer = CodeIndexer(tmp_path / "code.sqlite", [tmp_path])
    repo_id = indexer.index(str(root), "demo")["repo_id"]
    store = CodeStore(tmp_path / "code.sqlite")
    assert store.search("old_func")
    assert store.search("keep_func")

    parsed: list[str | None] = []
    real_parse = indexer_module.parse_file

    def spy(path, *, path_label=None):
        parsed.append(path_label)
        return real_parse(path, path_label=path_label)

    monkeypatch.setattr(indexer_module, "parse_file", spy)
    _write(root / "a.py", "def new_func():\n    return 3\n")
    second = indexer.index(str(root), "demo")
    assert parsed == ["a.py"]  # 只重解析变化的文件
    assert second["repo_id"] == repo_id
    assert second["files"] == 2
    assert store.search("old_func") == []
    assert store.search("new_func")
    assert store.search("keep_func")

    (root / "a.py").unlink()
    third = indexer.index(str(root), "demo")
    assert third["files"] == 1
    assert store.get_file_hash(repo_id, "a.py") is None
    assert store.search("new_func") == []
    overview = store.overview(repo_id)
    assert overview["file_count"] == 1


def test_reindex_clears_and_rebuilds_deterministically(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    indexer = CodeIndexer(tmp_path / "code.sqlite", [tmp_path])
    first = indexer.index(str(root), "demo")
    store = CodeStore(tmp_path / "code.sqlite")
    ids_before = {hit["chunk_id"] for hit in store.search("class def return", k=100)}
    assert ids_before

    second = indexer.index(str(root), "demo", reindex=True)
    ids_after = {hit["chunk_id"] for hit in store.search("class def return", k=100)}
    assert second["repo_id"] == first["repo_id"]
    assert {key: second[key] for key in ("files", "symbols", "imports", "references", "chunks")} == {
        key: first[key] for key in ("files", "symbols", "imports", "references", "chunks")
    }
    assert ids_after == ids_before


def test_allowed_roots_and_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    _write(outside / "a.py", "A = 1\n")
    db = tmp_path / "code.sqlite"
    indexer = CodeIndexer(db, [allowed])

    with pytest.raises(RootNotAllowedError):
        indexer.index(str(outside), "outside")
    link = allowed / "linked"
    os.symlink(outside, link)
    with pytest.raises(RootNotAllowedError):
        indexer.index(str(link), "linked")
    assert CodeStore(db).list_repos() == []


def test_scanner_limits_are_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "ok.py", "x = 1\n")
    _write(root / "long.py", "".join(f"v{i} = {i}\n" for i in range(6)))
    _write(root / "big.py", "y = 2\n" * 40)
    monkeypatch.setattr(scanner_module, "MAX_FILE_LINES", 4)
    monkeypatch.setattr(scanner_module, "MAX_FILE_BYTES", 80)

    report = CodeIndexer(tmp_path / "code.sqlite", [tmp_path]).index(str(root), "demo")
    reasons = {item["path"]: item["reason"] for item in report["skipped"]}
    assert reasons["long.py"] == REASON_TOO_MANY_LINES
    assert reasons["big.py"] == REASON_TOO_LARGE
    assert report["files"] == 1


def test_syntax_error_file_is_retried_and_reported(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "bad.py", "def broken(:\n")
    indexer = CodeIndexer(tmp_path / "code.sqlite", [tmp_path])
    first = indexer.index(str(root), "demo")
    second = indexer.index(str(root), "demo")
    assert first["files"] == second["files"] == 0
    assert first["errors"] and second["errors"]
    assert first["errors"][0]["path"] == "bad.py"
    assert "SyntaxError" in first["errors"][0]["error"]


def test_chunker_splits_long_symbol_deterministically() -> None:
    lines = ["def long_func():"]
    lines.extend(f"    value_{index} = {index}" for index in range(MAX_CHUNK_LINES + 10))
    source = "\n".join(lines) + "\n"
    info = parse_source("long.py", source)
    symbol = next(item for item in info.symbols if item.qualname == "long_func")
    assert symbol.end_line - symbol.start_line + 1 > MAX_CHUNK_LINES

    chunks = build_chunks("repo_demo", "long.py", info.symbols)
    assert len(chunks) >= 2
    assert chunks[0]["qualname"].endswith("#part1")
    assert chunks[1]["qualname"].endswith("#part2")
    assert chunks[0]["start_line"] == symbol.start_line
    for chunk in chunks:
        assert chunk["text"]
        assert len(chunk["text"]) <= MAX_CHUNK_CHARS
        assert chunk["end_line"] - chunk["start_line"] + 1 <= MAX_CHUNK_LINES
    assert [item["chunk_id"] for item in build_chunks("repo_demo", "long.py", info.symbols)] == [
        item["chunk_id"] for item in chunks
    ]


def test_empty_allowed_roots_denies_all(tmp_path: Path) -> None:
    """显式 [] 与 API 路由一致 = 全禁；None 才回退 home。"""
    root = _make_repo(tmp_path)
    db = tmp_path / "code.sqlite"
    indexer = CodeIndexer(db, [])
    with pytest.raises(RootNotAllowedError):
        indexer.index(str(root), "demo")
    assert CodeStore(db).list_repos() == []


def test_cli_default_db_uses_codeagent_data_root(tmp_path: Path) -> None:
    """CLI 不传 --db 时使用 $CODEAGENT_DATA_ROOT/code_index.sqlite。"""
    root = _make_repo(tmp_path)
    data_root = tmp_path / "data-root"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "CODEAGENT_DATA_ROOT": str(data_root),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "index_code.py"),
            "--root",
            str(root),
            "--name",
            "cli-default-db",
            "--allow-root",
            str(tmp_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert set(report) == REPORT_KEYS
    assert (data_root / "code_index.sqlite").is_file()


def test_cli_json_and_reindex(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    db = tmp_path / "cli.sqlite"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "index_code.py"),
            "--root",
            str(root),
            "--name",
            "cli-demo",
            "--db",
            str(db),
            "--allow-root",
            str(tmp_path),
            "--reindex",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert set(report) == REPORT_KEYS
    assert report["files"] == 3
    assert report["repo_id"].startswith("repo_")
