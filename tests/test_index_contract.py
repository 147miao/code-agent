"""W1 新增契约：trigram 中文、确定性、改/删清理、重依赖隔离、RRF。"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from codeagent.index.chunker import make_chunk_id
from codeagent.index.indexer import CodeIndexer
from codeagent.index.parser import parse_source
from codeagent.index.scanner import file_hash
from codeagent.index.store import CodeStore, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]

CHINESE_SOURCE = '''"""会话历史与确认令牌的中文说明。"""


def load_history():
    """从数据库读取会话历史，并校验确认令牌。"""
    return 1
'''


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _chunk_ids(db_path: Path, repo_id: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT chunk_id FROM chunks WHERE repo_id = ? ORDER BY chunk_id", (repo_id,)
            )
        ]
    finally:
        conn.close()


def test_trigram_chinese_hits_and_filters(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "pkg" / "history.py", CHINESE_SOURCE)
    db = tmp_path / "code.sqlite"
    report = CodeIndexer(db, [tmp_path]).index(str(root), "demo")
    store = CodeStore(db)
    repo_id = report["repo_id"]

    for query in ("会话历史", "确认令牌"):
        hits = store.search(query, k=5, repo_id=repo_id)
        assert [hit["path"] for hit in hits] == ["pkg/history.py"]
        assert hits[0]["qualname"] == "load_history"
        assert hits[0]["score"] > 0

    assert store.search("会话历史", k=5, path_prefix="pkg")
    assert store.search("会话历史", k=5, kind="function")
    assert store.search("会话历史", k=5, path_prefix="other") == []
    assert store.search("会话历史", k=5, kind="class") == []

    # 证明命中来自 trigram 表：unicode61 把整段中文当一个词，无法匹配子串。
    conn = sqlite3.connect(db)
    try:
        unicode_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ('"会话历史"',)
        ).fetchone()[0]
        trigram_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts_trigram WHERE chunks_fts_trigram MATCH ?",
            ('"会话历史"',),
        ).fetchone()[0]
    finally:
        conn.close()
    assert unicode_count == 0
    assert trigram_count >= 1


def test_chunk_id_and_file_hash_are_deterministic(tmp_path: Path) -> None:
    expected_payload = b"repo_demo|pkg/a.py|func|1|3"
    assert make_chunk_id("repo_demo", "pkg/a.py", "func", 1, 3) == hashlib.sha1(
        expected_payload
    ).hexdigest()

    root = tmp_path / "repo"
    source = "def f():\n    return 1\n"
    target = _write(root / "pkg" / "a.py", source)
    assert file_hash(target) == hashlib.sha1(source.encode("utf-8")).hexdigest()

    db = tmp_path / "code.sqlite"
    indexer = CodeIndexer(db, [tmp_path])
    first = indexer.index(str(root), "demo")
    first_ids = _chunk_ids(db, first["repo_id"])
    first_hashes = {item["path"]: item["file_hash"] for item in CodeStore(db).list_files(first["repo_id"])}
    assert first_ids and first_hashes["pkg/a.py"] == file_hash(target)

    second = indexer.index(str(root), "demo", reindex=True)
    assert second["repo_id"] == first["repo_id"]
    assert _chunk_ids(db, second["repo_id"]) == first_ids
    second_hashes = {item["path"]: item["file_hash"] for item in CodeStore(db).list_files(second["repo_id"])}
    assert second_hashes == first_hashes

    info = parse_source("pkg/a.py", source)
    from codeagent.index.chunker import build_chunks

    chunks = build_chunks(first["repo_id"], "pkg/a.py", info.symbols)
    assert all(
        chunk["chunk_id"]
        == make_chunk_id(
            first["repo_id"], chunk["path"], chunk["qualname"], chunk["start_line"], chunk["end_line"]
        )
        for chunk in chunks
    )


def test_modify_and_delete_cleanup_all_tables(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    old_source = "import os\n\ndef old_func():\n    return os.getcwd()\n"
    _write(root / "a.py", old_source)
    _write(root / "b.py", "class Keep:\n    def run(self):\n        return 1\n")
    db = tmp_path / "code.sqlite"
    indexer = CodeIndexer(db, [tmp_path])
    repo_id = indexer.index(str(root), "demo")["repo_id"]
    store = CodeStore(db)
    assert store.search("old_func")
    assert store.search("Keep")

    _write(root / "a.py", "def new_func():\n    return 3\n")
    (root / "b.py").unlink()
    report = indexer.index(str(root), "demo")
    assert report["files"] == 1
    assert store.search("old_func") == []
    assert store.search("new_func")
    assert store.search("Keep") == []
    assert store.get_file_hash(repo_id, "a.py") is not None
    assert store.get_file_hash(repo_id, "b.py") is None

    conn = sqlite3.connect(db)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE repo_id = ? AND path = 'a.py'", (repo_id,)
        ).fetchone()[0] > 0
        for table, column in (
            ("symbols", "path"),
            ("imports", "src_path"),
            ("references", "src_path"),
            ("chunks", "path"),
        ):
            quoted = f'"{table}"' if table == "references" else table
            count = conn.execute(
                f"SELECT COUNT(*) FROM {quoted} WHERE repo_id = ? AND {column} = ?", (repo_id, "b.py")
            ).fetchone()[0]
            assert count == 0, table
    finally:
        conn.close()


def test_import_codeagent_does_not_load_heavy_dependencies() -> None:
    script = (
        "import sys\n"
        "import codeagent\n"
        "import codeagent.citations\n"
        "import codeagent.index\n"
        "import codeagent.retrieval\n"
        "forbidden = ('torch', 'transformers', 'pymilvus', 'milvus', 'ragm', 'langgraph', 'langchain')\n"
        "loaded = [name for name in forbidden if name in sys.modules]\n"
        "if loaded:\n"
        "    print('loaded:', loaded)\n"
        "    raise SystemExit(1)\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


LEGACY_SCHEMA = """
CREATE TABLE chunks (
    repo_id TEXT NOT NULL,
    chunk_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    path, qualname, text, repo_id UNINDEXED, chunk_id UNINDEXED, kind UNINDEXED,
    tokenize = 'unicode61'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, path, qualname, text, repo_id, chunk_id, kind)
    VALUES (new.rowid, new.path, new.qualname, new.text, new.repo_id, new.chunk_id, new.kind);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.rowid;
    INSERT INTO chunks_fts(rowid, path, qualname, text, repo_id, chunk_id, kind)
    VALUES (new.rowid, new.path, new.qualname, new.text, new.repo_id, new.chunk_id, new.kind);
END;
"""


def test_legacy_single_fts_database_is_upgraded(tmp_path: Path) -> None:
    """旧库只有 unicode61：打开时应补建 trigram、替换触发器并回填旧行。"""
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(LEGACY_SCHEMA)
        conn.execute(
            "INSERT INTO chunks (repo_id, chunk_id, path, qualname, kind, start_line, end_line, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("repo_legacy", "legacy1", "pkg/legacy.py", "load", "function", 1, 2, "读取会话历史。"),
        )
        conn.commit()
    finally:
        conn.close()

    store = CodeStore(db)
    assert [hit["chunk_id"] for hit in store.search("会话历史")] == ["legacy1"]

    repo_id = store.add_repo(str(tmp_path / "repo"), "legacy")["repo_id"]
    store.insert_chunks(
        repo_id,
        [
            {
                "chunk_id": "new1",
                "path": "pkg/new.py",
                "qualname": "verify",
                "kind": "function",
                "start_line": 1,
                "end_line": 2,
                "text": "校验确认令牌。",
            }
        ],
    )
    assert [hit["chunk_id"] for hit in store.search("确认令牌")] == ["new1"]

    conn = sqlite3.connect(db)
    try:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("chunks", "chunks_fts", "chunks_fts_trigram")
        }
    finally:
        conn.close()
    assert counts == {"chunks": 2, "chunks_fts": 2, "chunks_fts_trigram": 2}


def _hit(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "path": f"{chunk_id}.py",
        "start_line": 1,
        "end_line": 1,
        "text": chunk_id,
    }


def test_reciprocal_rank_fusion_merges_two_lists() -> None:
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b")], [_hit("b"), _hit("c")]], k=3, rrf_k=60)
    assert [hit["chunk_id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert "rank" not in fused[0]

    assert reciprocal_rank_fusion([], 3) == []
    assert reciprocal_rank_fusion([[_hit("a")]], 0) == []
    assert [hit["chunk_id"] for hit in reciprocal_rank_fusion([[_hit("a")], []], 3)] == ["a"]
