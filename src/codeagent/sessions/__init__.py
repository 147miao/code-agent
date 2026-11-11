"""会话层：SQLite 默认 + MySQL 适配器 + factory。"""

from __future__ import annotations

from .repository import (
    DEFAULT_HISTORY_CHARS,
    DEFAULT_HISTORY_TURNS,
    SessionNotFoundError,
    SessionRepository,
    SQLiteSessionRepository,
    build_session_repository,
)

__all__ = [
    "DEFAULT_HISTORY_CHARS",
    "DEFAULT_HISTORY_TURNS",
    "SQLiteSessionRepository",
    "SessionNotFoundError",
    "SessionRepository",
    "build_session_repository",
]
