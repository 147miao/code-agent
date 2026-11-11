"""MySQL 会话仓储（pymysql）。

DSN 形如 mysql+pymysql://user:pass@127.0.0.1:3307/codeagent?charset=utf8mb4；
每次操作独立连接（与 SQLite 实现一致），表结构/返回字段保持一致。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import unquote, urlsplit

import pymysql
from pymysql.cursors import DictCursor

from .repository import (
    DEFAULT_HISTORY_CHARS,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    MAX_TITLE_CHARS,
    SessionNotFoundError,
    build_history,
    new_session_id,
    session_dict,
    utcnow,
    validate_message,
)

DEFAULT_CONNECT_TIMEOUT = 5.0
MYSQL_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id VARCHAR(64) NOT NULL PRIMARY KEY,
        title VARCHAR(255) NOT NULL DEFAULT '',
        created_at VARCHAR(32) NOT NULL,
        updated_at VARCHAR(32) NOT NULL,
        KEY idx_sessions_updated (updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(64) NOT NULL,
        role VARCHAR(16) NOT NULL,
        content MEDIUMTEXT NOT NULL,
        meta_json TEXT NOT NULL,
        created_at VARCHAR(32) NOT NULL,
        KEY idx_messages_session (session_id, id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
)

__all__ = ["MySQLSessionRepository", "parse_mysql_dsn"]


def parse_mysql_dsn(dsn: str) -> dict[str, Any]:
    """解析 mysql+pymysql:// DSN；非法 scheme / 缺库名时抛 ValueError。"""
    parts = urlsplit(str(dsn or "").strip())
    if parts.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"不支持的 MySQL DSN scheme: {parts.scheme}")
    database = parts.path.lstrip("/")
    if not database:
        raise ValueError("MySQL DSN 缺少数据库名")
    query = {
        key: value
        for key, value in (
            item.split("=", 1) if "=" in item else (item, "") for item in parts.query.split("&")
        )
        if key
    }
    return {
        "host": parts.hostname or "127.0.0.1",
        "port": int(parts.port or 3306),
        "user": unquote(parts.username or ""),
        "password": unquote(parts.password or ""),
        "database": database,
        "charset": query.get("charset") or "utf8mb4",
    }


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MySQLSessionRepository:
    """MySQL 版会话仓储；契约与 SQLiteSessionRepository 一致。"""

    def __init__(self, dsn: str, *, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        self.dsn = str(dsn or "").strip()
        self.connect_timeout = float(connect_timeout)
        self.params = parse_mysql_dsn(self.dsn)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[pymysql.connections.Connection]:
        conn = pymysql.connect(
            host=self.params["host"],
            port=self.params["port"],
            user=self.params["user"],
            password=self.params["password"],
            database=self.params["database"],
            charset=self.params["charset"],
            cursorclass=DictCursor,
            connect_timeout=self.connect_timeout,
            autocommit=False,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn, conn.cursor() as cursor:
            for statement in MYSQL_SCHEMA:
                cursor.execute(statement)

    def create_session(self, title: str = "") -> dict[str, Any]:
        session_id = new_session_id()
        now = utcnow()
        clean_title = str(title or "").strip()[:MAX_TITLE_CHARS]
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)",
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
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT * FROM sessions WHERE id = %s", (sid,))
            row = cursor.fetchone()
        return session_dict(row) if row is not None else None

    def list_sessions(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict[str, Any]]:
        count = max(1, min(_to_int(limit, DEFAULT_LIST_LIMIT), MAX_LIST_LIMIT))
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC, id DESC LIMIT %s", (count,)
            )
            rows = list(cursor.fetchall())
        return [session_dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        sid = str(session_id or "").strip()
        if not sid:
            return False
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM sessions WHERE id = %s", (sid,))
            if cursor.fetchone() is None:
                return False
            cursor.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
            cursor.execute("DELETE FROM sessions WHERE id = %s", (sid,))
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
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM sessions WHERE id = %s", (sid,))
            if cursor.fetchone() is None:
                raise SessionNotFoundError(f"session 不存在: {sid}")
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, meta_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (sid, role_text, text, meta_json, now),
            )
            message_id = int(cursor.lastrowid or 0)
            cursor.execute("UPDATE sessions SET updated_at = %s WHERE id = %s", (now, sid))
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
        limit_turns = max(0, _to_int(max_turns, DEFAULT_HISTORY_TURNS))
        if not sid or limit_turns == 0:
            return []
        with self._connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = %s ORDER BY id ASC", (sid,)
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return build_history(rows, max_turns=max_turns, max_chars=max_chars)
