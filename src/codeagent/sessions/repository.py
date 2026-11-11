"""会话仓储：SessionRepository 契约 + SQLite 默认实现 + factory。

设计：
- SQLite 用 WAL + busy_timeout，每次操作独立连接（适配 FastAPI 线程池并发）；
- 表结构：sessions(id,title,created_at,updated_at)、messages(id,session_id,role,content,meta_json,created_at)；
- get_history 支持按轮数与字符数截断（从最新往回取，保持时间正序）；
- DATABASE_URL 为空时用 SQLite；mysql+pymysql://... 时在 factory 中惰性导入 MySQL 实现。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..config import Settings

BUSY_TIMEOUT_MS = 3000
DEFAULT_HISTORY_TURNS = 6
DEFAULT_HISTORY_CHARS = 4000
MAX_MESSAGE_CHARS = 20000
MAX_TITLE_CHARS = 200
MAX_LIST_LIMIT = 200
DEFAULT_LIST_LIMIT = 50
ROLES = ("user", "assistant", "system")
SESSION_ID_PREFIX = "sess_"

SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
"""

__all__ = [
    "DEFAULT_HISTORY_CHARS",
    "DEFAULT_HISTORY_TURNS",
    "DEFAULT_LIST_LIMIT",
    "MAX_MESSAGE_CHARS",
    "SQLiteSessionRepository",
    "SessionNotFoundError",
    "SessionRepository",
    "build_history",
    "build_session_repository",
    "new_session_id",
    "parse_meta",
    "validate_message",
]


class SessionNotFoundError(LookupError):
    """session_id 不存在。"""


class SessionRepository(Protocol):
    """会话仓储契约（SQLite / MySQL 两种实现必须一致）。"""

    def create_session(self, title: str = "") -> dict[str, Any]:
        ...

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        ...

    def list_sessions(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict[str, Any]]:
        ...

    def delete_session(self, session_id: str) -> bool:
        ...

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def get_history(
        self,
        session_id: str,
        max_turns: int = DEFAULT_HISTORY_TURNS,
        max_chars: int = DEFAULT_HISTORY_CHARS,
    ) -> list[dict[str, Any]]:
        ...


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def new_session_id() -> str:
    return SESSION_ID_PREFIX + uuid.uuid4().hex[:16]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """同时兼容 Mapping 与 sqlite3.Row（后者没有 .get）。"""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def parse_meta(raw: Any) -> dict[str, Any]:
    """meta_json 解析；损坏/空值返回空 dict。"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def session_dict(row: Any) -> dict[str, Any]:
    return {
        "id": str(_row_get(row, "id") or ""),
        "title": str(_row_get(row, "title") or ""),
        "created_at": str(_row_get(row, "created_at") or ""),
        "updated_at": str(_row_get(row, "updated_at") or ""),
    }


def message_dict(row: Any) -> dict[str, Any]:
    return {
        "id": _to_int(_row_get(row, "id")),
        "session_id": str(_row_get(row, "session_id") or ""),
        "role": str(_row_get(row, "role") or ""),
        "content": str(_row_get(row, "content") or ""),
        "meta": parse_meta(_row_get(row, "meta_json")),
        "created_at": str(_row_get(row, "created_at") or ""),
    }


def validate_message(role: str, content: str) -> tuple[str, str]:
    role_text = str(role or "").strip()
    if role_text not in ROLES:
        raise ValueError(f"非法 role: {role!r}（允许 {', '.join(ROLES)}）")
    text = str(content or "")
    if not text.strip():
        raise ValueError("content 不能为空")
    return role_text, text[:MAX_MESSAGE_CHARS]


def build_history(
    rows: list[dict[str, Any]],
    *,
    max_turns: int = DEFAULT_HISTORY_TURNS,
    max_chars: int = DEFAULT_HISTORY_CHARS,
) -> list[dict[str, Any]]:
    """把消息行按轮数/字数截断为时间正序历史。

    rows 必须按时间正序；先取最近 max_turns*2 条，再从最新往回累计字符预算，
    单条超预算时截断内容（保留开头），最后恢复正序。
    """
    limit = max(0, _to_int(max_turns, DEFAULT_HISTORY_TURNS)) * 2
    if limit == 0:
        return []
    budget = max(0, _to_int(max_chars, DEFAULT_HISTORY_CHARS))
    selected: list[dict[str, Any]] = []
    used = 0
    for row in reversed(rows[-limit:]):
        content = str(_row_get(row, "content") or "")
        if budget and used + len(content) > budget:
            remaining = budget - used
            if remaining <= 0:
                break
            content = content[:remaining]
        item = message_dict(row)
        item["content"] = content
        selected.append(item)
        used += len(content)
        if budget and used >= budget:
            break
    selected.reverse()
    return selected


@contextmanager
def _sqlite_connection(db_path: Path, busy_timeout_ms: int) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    try:
        yield conn
    finally:
        conn.close()


class SQLiteSessionRepository:
    """默认实现：WAL + busy_timeout，每次操作独立连接。"""

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self.db_path = Path(db_path).expanduser()
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _sqlite_connection(self.db_path, self.busy_timeout_ms) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SESSION_SCHEMA)
            conn.commit()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with _sqlite_connection(self.db_path, self.busy_timeout_ms) as conn:
            yield conn

    def create_session(self, title: str = "") -> dict[str, Any]:
        session_id = new_session_id()
        now = utcnow()
        clean_title = str(title or "").strip()[:MAX_TITLE_CHARS]
        with self._connection() as conn, conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, clean_title, now, now),
            )
        return {
            "id": session_id,
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        return session_dict(row) if row is not None else None

    def list_sessions(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict[str, Any]]:
        count = max(1, min(_to_int(limit, DEFAULT_LIST_LIMIT), MAX_LIST_LIMIT))
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC, id DESC LIMIT ?", (count,)
            ).fetchall()
        return [session_dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        sid = str(session_id or "").strip()
        if not sid:
            return False
        with self._connection() as conn, conn:
            existing = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone()
            if existing is None:
                return False
            conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        return True

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        sid = str(session_id or "").strip()
        role_text, text = validate_message(role, content)
        now = utcnow()
        meta_json = json.dumps(dict(meta or {}), ensure_ascii=False, default=str)
        with self._connection() as conn, conn:
            existing = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone()
            if existing is None:
                raise SessionNotFoundError(f"session 不存在: {sid}")
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, meta_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, role_text, text, meta_json, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
            message_id = int(cursor.lastrowid or 0)
        return {
            "id": message_id,
            "session_id": sid,
            "role": role_text,
            "content": text,
            "meta": dict(meta or {}),
            "created_at": now,
        }

    def get_history(
        self,
        session_id: str,
        max_turns: int = DEFAULT_HISTORY_TURNS,
        max_chars: int = DEFAULT_HISTORY_CHARS,
    ) -> list[dict[str, Any]]:
        sid = str(session_id or "").strip()
        limit = max(0, _to_int(max_turns, DEFAULT_HISTORY_TURNS)) * 2
        if not sid or limit == 0:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (sid,)
            ).fetchall()
        return build_history(
            [dict(row) for row in rows], max_turns=max_turns, max_chars=max_chars
        )


def build_session_repository(settings: Settings) -> SessionRepository:
    """按 DATABASE_URL 选择实现：空 = SQLite，mysql+pymysql:// = MySQL。"""
    url = str(settings.database_url or "").strip()
    if not url:
        return SQLiteSessionRepository(settings.session_store_path)
    if url.startswith("mysql"):
        from .mysql import MySQLSessionRepository

        return MySQLSessionRepository(url)
    raise ValueError(f"不支持的 DATABASE_URL: {url.split(':', 1)[0]}")
