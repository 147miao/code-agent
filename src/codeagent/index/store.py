"""SQLite + FTS5 持久化（代码索引 v2）。

设计：
- WAL、synchronous=NORMAL、busy_timeout=3000ms；每次操作使用独立连接，适配 API 线程并发；
- 6 张表按 plan 冻结 schema：repos/files/symbols/imports/references/chunks；
- 双 FTS5 表：chunks_fts（unicode61，关键词/标识符）与 chunks_fts_trigram（trigram，
  中文子串），均由 chunks 上的触发器同步；
- search 分别查询两张 FTS 表，再用 RRF（Reciprocal Rank Fusion）合并；
- 所有过滤值走 SQL 参数，不拼接用户输入；
- 所有 `repo_id` 维度的方法在 repo 不存在时返回空/False；调用方负责 404；
- reverse_imports 是文件级启发式：用 import 的 module 与目标文件路径的模块名后缀匹配，
  不承诺精确调用链。
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import SCHEMA, TRIGGERS

BUSY_TIMEOUT_MS = 3000
MAX_SEARCH_K = 100
MAX_FTS_TERMS = 16
MAX_FTS_TERM_CHARS = 128
TRIGRAM_MIN_CHARS = 3
RRF_K = 60
RRF_CANDIDATE_FACTOR = 4
RRF_MIN_CANDIDATES = 20
FTS_WORD_RE = re.compile(r"\w+", re.UNICODE)
LIKE_ESCAPE = "!"
TOP_PATH_LIMIT = 10


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_root(root: str | Path) -> str:
    return str(Path(root).expanduser().resolve())


def _escape_like(value: str) -> str:
    return value.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2).replace("%", LIKE_ESCAPE + "%").replace(
        "_", LIKE_ESCAPE + "_"
    )


def _query_terms(query: str) -> list[str]:
    """提取查询词元（Unicode 字母/数字/下划线），去重并限制数量与单项长度。"""
    terms: list[str] = []
    for match in FTS_WORD_RE.finditer(query or ""):
        term = match.group(0)[:MAX_FTS_TERM_CHARS]
        if term not in terms:
            terms.append(term)
        if len(terms) >= MAX_FTS_TERMS:
            break
    return terms


def _fts_query(query: str) -> str | None:
    """unicode61 查询：任意用户输入清洗为逐项加引号的 OR 查询。"""
    terms = _query_terms(query)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def _trigram_query(query: str) -> str | None:
    """trigram 查询：只保留长度 >=3 的词元（trigram 索引无法匹配更短子串）。"""
    terms = [term for term in _query_terms(query) if len(term) >= TRIGRAM_MIN_CHARS]
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def _module_candidates(path: str) -> set[str]:
    """由仓库相对路径推断可能的 Python 模块名（所有点号后缀）。"""
    normalized = str(path).replace("\\", "/").lstrip("./")
    normalized = normalized.removesuffix(".py")
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return {".".join(parts[index:]) for index in range(len(parts))}


def _ensure_dual_fts_triggers(conn: sqlite3.Connection) -> None:
    """把旧库只有 unicode61 的触发器升级为双表触发器。

    `CREATE TRIGGER IF NOT EXISTS` 不会覆盖同名旧触发器，所以这里先检测 `chunks_ai` 的
    SQL；只有确认缺少 trigram 写入时才重建，避免每次打开数据库都做 DDL。
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'chunks_ai'"
    ).fetchone()
    if row is not None and "chunks_fts_trigram" in str(row[0] or ""):
        return
    for name in ("chunks_ai", "chunks_ad", "chunks_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.executescript(TRIGGERS)


def _ensure_trigram_synced(conn: sqlite3.Connection) -> None:
    """旧库首次升级时回填 trigram 表（触发器只覆盖新写入的行）。"""
    chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    trigram_count = int(conn.execute("SELECT COUNT(*) FROM chunks_fts_trigram").fetchone()[0])
    if chunk_count == trigram_count:
        return
    conn.execute("DELETE FROM chunks_fts_trigram")
    conn.execute(
        "INSERT INTO chunks_fts_trigram (rowid, path, qualname, text, repo_id, chunk_id, kind) "
        "SELECT rowid, path, qualname, text, repo_id, chunk_id, kind FROM chunks"
    )


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[dict[str, Any]]],
    k: int,
    *,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """把多路有序检索结果按 RRF 合并：score = Σ 1/(rrf_k + rank)。

    同一 chunk_id 只保留一条，字段取首次出现的值，score 为各路的 RRF 之和；
    排序为 score 降序，并用 path/start_line/chunk_id 保证确定性。
    """
    limit = int(k)
    if limit <= 0:
        return []
    weight_base = max(1, int(rrf_k))
    fused: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            chunk_id = str(hit.get("chunk_id") or "")
            if not chunk_id:
                continue
            entry = fused.get(chunk_id)
            if entry is None:
                entry = {key: value for key, value in hit.items() if key != "rank"}
                entry["score"] = 0.0
                fused[chunk_id] = entry
            entry["score"] = float(entry["score"]) + 1.0 / (weight_base + rank)
    ranked = sorted(
        fused.values(),
        key=lambda hit: (
            -float(hit.get("score") or 0.0),
            str(hit.get("path") or ""),
            int(hit.get("start_line") or 0),
            str(hit.get("chunk_id") or ""),
        ),
    )
    return ranked[:limit]


class CodeStore:
    """代码索引 SQLite 存储；线程安全（每次操作独立连接）。"""

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self.db_path = Path(db_path)
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.executescript(SCHEMA)
                _ensure_dual_fts_triggers(conn)
                _ensure_trigram_synced(conn)
                conn.commit()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "fts5" in message or "trigram" in message:
                raise RuntimeError("当前 SQLite 未启用 FTS5/trigram，无法创建代码索引") from exc
            raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=self.busy_timeout_ms / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        try:
            yield conn
        finally:
            conn.close()

    # ---------- repos ----------

    def add_repo(self, root: str, name: str) -> dict[str, Any]:
        """按 root 幂等注册仓库：已存在则更新 name，返回当前统计。"""
        normalized = _normalize_root(root)
        repo_id = f"repo_{_sha1(normalized)[:16]}"
        now = _utcnow()
        with self._connection() as conn, conn:
            conn.execute(
                "INSERT INTO repos (id, root, name, created_at, updated_at, file_count, symbol_count, status) "
                "VALUES (?, ?, ?, ?, ?, 0, 0, 'pending') "
                "ON CONFLICT(root) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at",
                (repo_id, normalized, name or Path(normalized).name, now, now),
            )
            row = conn.execute("SELECT * FROM repos WHERE root = ?", (normalized,)).fetchone()
        result = _repo_row(row)
        if result is None:  # pragma: no cover - upsert 后必然存在
            raise RuntimeError(f"repo disappeared after upsert: {normalized}")
        return result

    def list_repos(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM repos ORDER BY updated_at DESC, name ASC").fetchall()
        return [_repo_row(row) for row in rows]  # type: ignore[misc]

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
        return _repo_row(row)

    def delete_repo(self, repo_id: str) -> bool:
        with self._connection() as conn, conn:
            existing = conn.execute("SELECT 1 FROM repos WHERE id = ?", (repo_id,)).fetchone()
            if existing is None:
                return False
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM symbols WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM imports WHERE repo_id = ?", (repo_id,))
            conn.execute('DELETE FROM "references" WHERE repo_id = ?', (repo_id,))
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM repos WHERE id = ?", (repo_id,))
        return True

    def reindex_repo(self, repo_id: str) -> bool:
        """清空仓库所有派生行（事务），保留 repos 行；用于 --reindex。"""
        now = _utcnow()
        with self._connection() as conn, conn:
            existing = conn.execute("SELECT 1 FROM repos WHERE id = ?", (repo_id,)).fetchone()
            if existing is None:
                return False
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM symbols WHERE repo_id = ?", (repo_id,))
            conn.execute("DELETE FROM imports WHERE repo_id = ?", (repo_id,))
            conn.execute('DELETE FROM "references" WHERE repo_id = ?', (repo_id,))
            conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            conn.execute(
                "UPDATE repos SET file_count = 0, symbol_count = 0, status = 'pending', updated_at = ? "
                "WHERE id = ?",
                (now, repo_id),
            )
        return True

    def update_repo_stats(
        self,
        repo_id: str,
        *,
        file_count: int | None = None,
        symbol_count: int | None = None,
        status: str | None = None,
    ) -> bool:
        sets = ["updated_at = ?"]
        params: list[Any] = [_utcnow()]
        if file_count is not None:
            sets.append("file_count = ?")
            params.append(int(file_count))
        if symbol_count is not None:
            sets.append("symbol_count = ?")
            params.append(int(symbol_count))
        if status is not None:
            sets.append("status = ?")
            params.append(str(status))
        params.append(repo_id)
        with self._connection() as conn, conn:
            cursor = conn.execute(f"UPDATE repos SET {', '.join(sets)} WHERE id = ?", params)
        return cursor.rowcount == 1

    # ---------- files ----------

    def get_file_hash(self, repo_id: str, path: str) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT file_hash FROM files WHERE repo_id = ? AND path = ?", (repo_id, path)
            ).fetchone()
        return str(row["file_hash"]) if row is not None else None

    def list_files(self, repo_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT path, file_hash, size, mtime FROM files WHERE repo_id = ? ORDER BY path ASC",
                (repo_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_file(
        self,
        repo_id: str,
        path: str,
        file_hash: str,
        size: int,
        mtime: float,
    ) -> None:
        with self._connection() as conn, conn:
            conn.execute(
                "INSERT INTO files (repo_id, path, file_hash, size, mtime) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(repo_id, path) DO UPDATE SET file_hash = excluded.file_hash, "
                "size = excluded.size, mtime = excluded.mtime",
                (repo_id, path, file_hash, int(size), float(mtime)),
            )

    def delete_file_rows(self, repo_id: str, path: str) -> None:
        """删除单个文件在 5 张派生表中的全部行（chunks 触发器同步两张 FTS 表）。"""
        with self._connection() as conn, conn:
            conn.execute("DELETE FROM chunks WHERE repo_id = ? AND path = ?", (repo_id, path))
            conn.execute("DELETE FROM symbols WHERE repo_id = ? AND path = ?", (repo_id, path))
            conn.execute("DELETE FROM imports WHERE repo_id = ? AND src_path = ?", (repo_id, path))
            conn.execute('DELETE FROM "references" WHERE repo_id = ? AND src_path = ?', (repo_id, path))
            conn.execute("DELETE FROM files WHERE repo_id = ? AND path = ?", (repo_id, path))

    # ---------- rows ----------

    def insert_symbols(self, repo_id: str, path: str, symbols: Sequence[Any]) -> None:
        rows = []
        for symbol in symbols:
            code = str(_value(symbol, "code") or "")
            rows.append(
                (
                    repo_id,
                    path,
                    str(_value(symbol, "qualname") or ""),
                    str(_value(symbol, "kind") or ""),
                    int(_value(symbol, "start_line") or 1),
                    int(_value(symbol, "end_line") or 1),
                    _value(symbol, "docstring"),
                    _sha1(code),
                    code,
                )
            )
        if not rows:
            return
        with self._connection() as conn, conn:
            conn.executemany(
                "INSERT INTO symbols "
                "(repo_id, path, qualname, kind, start_line, end_line, docstring, code_hash, code) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def insert_imports(self, repo_id: str, path: str, imports: Sequence[Any]) -> None:
        rows = [
            (
                repo_id,
                path,
                str(_value(item, "module") or ""),
                int(_value(item, "level") or 0),
                int(_value(item, "lineno") or 0),
            )
            for item in imports
        ]
        if not rows:
            return
        with self._connection() as conn, conn:
            conn.executemany(
                "INSERT INTO imports (repo_id, src_path, module, level, lineno) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def insert_references(self, repo_id: str, path: str, references: Sequence[Any]) -> None:
        rows = [
            (repo_id, path, str(_value(item, "name") or ""), int(_value(item, "lineno") or 0))
            for item in references
        ]
        if not rows:
            return
        with self._connection() as conn, conn:
            conn.executemany(
                'INSERT INTO "references" (repo_id, src_path, name, lineno) VALUES (?, ?, ?, ?)',
                rows,
            )

    def insert_chunks(self, repo_id: str, chunks: Sequence[Any]) -> None:
        rows = []
        for chunk in chunks:
            chunk_id = str(_value(chunk, "chunk_id") or "")
            if not chunk_id:
                raise ValueError("chunk 缺少 chunk_id")
            rows.append(
                (
                    repo_id,
                    chunk_id,
                    str(_value(chunk, "path") or ""),
                    str(_value(chunk, "qualname") or ""),
                    str(_value(chunk, "kind") or ""),
                    int(_value(chunk, "start_line") or 1),
                    int(_value(chunk, "end_line") or 1),
                    str(_value(chunk, "text") or ""),
                )
            )
        if not rows:
            return
        with self._connection() as conn, conn:
            conn.executemany(
                "INSERT INTO chunks "
                "(repo_id, chunk_id, path, qualname, kind, start_line, end_line, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    # ---------- search ----------

    def search(
        self,
        query: str,
        k: int = 8,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """双 FTS 检索：unicode61 + trigram 各取一批候选，RRF 合并后返回 top k。

        返回的 score 是 RRF 分（越大越相关）；path/kind/repo 过滤对两路查询都生效。
        """
        limit = int(k)
        if limit <= 0:
            return []
        limit = min(MAX_SEARCH_K, limit)
        candidate_limit = min(MAX_SEARCH_K, max(limit * RRF_CANDIDATE_FACTOR, RRF_MIN_CANDIDATES))
        branches: list[list[dict[str, Any]]] = []
        for table, match_query in (
            ("chunks_fts", _fts_query(query)),
            ("chunks_fts_trigram", _trigram_query(query)),
        ):
            if match_query is None:
                continue
            branches.append(
                self._search_fts(
                    table,
                    match_query,
                    candidate_limit,
                    repo_id=repo_id,
                    path_prefix=path_prefix,
                    kind=kind,
                )
            )
        if not branches:
            return []
        return reciprocal_rank_fusion(branches, limit)

    def _search_fts(
        self,
        table: str,
        match_query: str,
        limit: int,
        *,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """在指定 FTS 表上做单路检索；table 只能取内部常量。"""
        sql = [
            (
                "SELECT c.repo_id, c.chunk_id, c.path, c.qualname, c.kind, c.start_line, c.end_line, "
                f"c.text, bm25({table}) AS rank "
                f"FROM {table} JOIN chunks c ON c.rowid = {table}.rowid "
                f"WHERE {table} MATCH ?"
            )
        ]
        params: list[Any] = [match_query]
        if repo_id:
            sql.append("AND c.repo_id = ?")
            params.append(repo_id)
        if path_prefix:
            prefix = str(path_prefix).replace("\\", "/").rstrip("/")
            if prefix:
                sql.append("AND (c.path = ? OR c.path LIKE ? ESCAPE ?)")
                params.extend([prefix, _escape_like(prefix) + "/%", LIKE_ESCAPE])
        if kind:
            sql.append("AND c.kind = ?")
            params.append(kind)
        sql.append("ORDER BY rank ASC, c.path ASC, c.start_line ASC, c.chunk_id ASC LIMIT ?")
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [dict(row) for row in rows]

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        """按 qualname（可选 path）精确取符号；同 qualname 多定义时取最靠前者。"""
        sql = (
            "SELECT repo_id, path, qualname, kind, start_line, end_line, docstring, code_hash, code "
            "FROM symbols WHERE repo_id = ? AND qualname = ?"
        )
        params: list[Any] = [repo_id, qualname]
        if path is not None:
            sql += " AND path = ?"
            params.append(path)
        sql += " ORDER BY path ASC, start_line ASC LIMIT 1"
        with self._connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        """文件级反向 import（启发式）：返回 import 了该文件模块名的其他文件。"""
        relative = self._relative_path(repo_id, path)
        candidates = _module_candidates(relative)
        if not candidates:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT src_path, module FROM imports "
                "WHERE repo_id = ? AND src_path <> ? AND module <> ''",
                (repo_id, relative),
            ).fetchall()
        found: set[str] = set()
        for row in rows:
            module = str(row["module"])
            if any(module == candidate or module.endswith("." + candidate) for candidate in candidates):
                found.add(str(row["src_path"]))
        return sorted(found)

    def _relative_path(self, repo_id: str, path: str) -> str:
        normalized = str(path).replace("\\", "/")
        repo = self.get_repo(repo_id)
        if repo is not None and Path(normalized).is_absolute():
            try:
                return Path(normalized).relative_to(repo["root"]).as_posix()
            except ValueError:
                pass
        return normalized.lstrip("./")

    def overview(self, repo_id: str) -> dict[str, Any]:
        """仓库概览；repo 不存在返回空 dict。"""
        repo = self.get_repo(repo_id)
        if repo is None:
            return {}
        with self._connection() as conn:
            file_count = int(conn.execute("SELECT COUNT(*) FROM files WHERE repo_id = ?", (repo_id,)).fetchone()[0])
            import_count = int(
                conn.execute("SELECT COUNT(*) FROM imports WHERE repo_id = ?", (repo_id,)).fetchone()[0]
            )
            reference_count = int(
                conn.execute('SELECT COUNT(*) FROM "references" WHERE repo_id = ?', (repo_id,)).fetchone()[0]
            )
            chunk_count = int(
                conn.execute("SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,)).fetchone()[0]
            )
            kind_rows = conn.execute(
                "SELECT kind, COUNT(*) AS total FROM symbols WHERE repo_id = ? GROUP BY kind", (repo_id,)
            ).fetchall()
            path_rows = conn.execute(
                "SELECT path, COUNT(*) AS total FROM symbols WHERE repo_id = ? "
                "GROUP BY path ORDER BY total DESC, path ASC LIMIT ?",
                (repo_id, TOP_PATH_LIMIT),
            ).fetchall()
        by_kind = {str(row["kind"]): int(row["total"]) for row in kind_rows}
        return {
            **repo,
            "file_count": file_count,
            "symbol_count": sum(by_kind.values()),
            "module_count": by_kind.get("module", 0),
            "class_count": by_kind.get("class", 0),
            "function_count": by_kind.get("function", 0),
            "method_count": by_kind.get("method", 0),
            "kinds": by_kind,
            "top_paths": [
                {"path": str(row["path"]), "symbols": int(row["total"])} for row in path_rows
            ],
            "import_count": import_count,
            "reference_count": reference_count,
            "chunk_count": chunk_count,
        }


def _repo_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "repo_id": str(row["id"]),
        "root": str(row["root"]),
        "name": str(row["name"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "file_count": int(row["file_count"]),
        "symbol_count": int(row["symbol_count"]),
        "status": str(row["status"]),
    }
