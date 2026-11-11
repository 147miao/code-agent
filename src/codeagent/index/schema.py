"""SQLite schema：表、双 FTS5 表与同步触发器。

与 store.py 分离，避免单文件过长；SQL 全部使用 IF NOT EXISTS，旧库升级由
store._ensure_dual_fts_triggers / _ensure_trigram_synced 负责。
"""
from __future__ import annotations

_TABLES = """
CREATE TABLE IF NOT EXISTS repos (
    id TEXT PRIMARY KEY,
    root TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    file_count INTEGER NOT NULL DEFAULT 0,
    symbol_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_repos_root ON repos(root);

CREATE TABLE IF NOT EXISTS files (
    repo_id TEXT NOT NULL,
    path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    mtime REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);

CREATE TABLE IF NOT EXISTS symbols (
    repo_id TEXT NOT NULL,
    path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    docstring TEXT,
    code_hash TEXT NOT NULL,
    code TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_lookup ON symbols(repo_id, qualname, path);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(repo_id, path);

CREATE TABLE IF NOT EXISTS imports (
    repo_id TEXT NOT NULL,
    src_path TEXT NOT NULL,
    module TEXT NOT NULL,
    level INTEGER NOT NULL DEFAULT 0,
    lineno INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(repo_id, module);
CREATE INDEX IF NOT EXISTS idx_imports_src ON imports(repo_id, src_path);

CREATE TABLE IF NOT EXISTS "references" (
    repo_id TEXT NOT NULL,
    src_path TEXT NOT NULL,
    name TEXT NOT NULL,
    lineno INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_references_name ON "references"(repo_id, name);
CREATE INDEX IF NOT EXISTS idx_references_src ON "references"(repo_id, src_path);

CREATE TABLE IF NOT EXISTS chunks (
    repo_id TEXT NOT NULL,
    chunk_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(repo_id, path);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    path,
    qualname,
    text,
    repo_id UNINDEXED,
    chunk_id UNINDEXED,
    kind UNINDEXED,
    tokenize = 'unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
    path,
    qualname,
    text,
    repo_id UNINDEXED,
    chunk_id UNINDEXED,
    kind UNINDEXED,
    tokenize = 'trigram'
);
"""

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, path, qualname, text, repo_id, chunk_id, kind)
    VALUES (new.rowid, new.path, new.qualname, new.text, new.repo_id, new.chunk_id, new.kind);
    INSERT INTO chunks_fts_trigram(rowid, path, qualname, text, repo_id, chunk_id, kind)
    VALUES (new.rowid, new.path, new.qualname, new.text, new.repo_id, new.chunk_id, new.kind);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.rowid;
    DELETE FROM chunks_fts_trigram WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.rowid;
    DELETE FROM chunks_fts_trigram WHERE rowid = old.rowid;
    INSERT INTO chunks_fts(rowid, path, qualname, text, repo_id, chunk_id, kind)
    VALUES (new.rowid, new.path, new.qualname, new.text, new.repo_id, new.chunk_id, new.kind);
    INSERT INTO chunks_fts_trigram(rowid, path, qualname, text, repo_id, chunk_id, kind)
    VALUES (new.rowid, new.path, new.qualname, new.text, new.repo_id, new.chunk_id, new.kind);
END;
"""

SCHEMA = _TABLES + TRIGGERS

