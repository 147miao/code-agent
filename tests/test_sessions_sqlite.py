"""SQLite 会话仓储：CRUD、重启、历史截断、WAL、factory。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from codeagent.config import Settings
from codeagent.sessions.repository import (
    SessionNotFoundError,
    SQLiteSessionRepository,
    build_history,
    build_session_repository,
    parse_meta,
)


def test_crud_and_restart(tmp_path: Path) -> None:
    db = tmp_path / "sessions.sqlite"
    repo = SQLiteSessionRepository(db)
    session = repo.create_session("第一轮")
    assert session["id"].startswith("sess_")
    assert session["title"] == "第一轮"
    assert repo.get_session(session["id"])["title"] == "第一轮"
    assert [item["id"] for item in repo.list_sessions()] == [session["id"]]
    assert repo.get_session("missing") is None

    message = repo.append_message(session["id"], "user", "hello", meta={"k": 1})
    assert message["id"] > 0
    assert message["role"] == "user"
    repo.append_message(session["id"], "assistant", "world")
    history = repo.get_history(session["id"])
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["meta"] == {"k": 1}
    assert history[1]["session_id"] == session["id"]

    reopened = SQLiteSessionRepository(db)
    assert [item["content"] for item in reopened.get_history(session["id"])] == ["hello", "world"]
    assert reopened.delete_session(session["id"]) is True
    assert reopened.get_session(session["id"]) is None
    assert reopened.delete_session(session["id"]) is False
    assert reopened.get_history(session["id"]) == []


def test_wal_and_busy_timeout(tmp_path: Path) -> None:
    repo = SQLiteSessionRepository(tmp_path / "s.sqlite")
    with sqlite3.connect(str(repo.db_path)) as conn:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    with repo._connection() as conn:
        assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 3000


def test_history_truncation(tmp_path: Path) -> None:
    repo = SQLiteSessionRepository(tmp_path / "s.sqlite")
    sid = repo.create_session()["id"]
    for index in range(10):
        repo.append_message(sid, "user", f"u{index}")
        repo.append_message(sid, "assistant", f"a{index}")

    recent = repo.get_history(sid, max_turns=2)
    assert [item["content"] for item in recent] == ["u8", "a8", "u9", "a9"]

    capped = repo.get_history(sid, max_turns=3, max_chars=6)
    assert capped
    assert [item["content"] for item in capped] == ["a8", "u9", "a9"]
    assert sum(len(item["content"]) for item in capped) <= 6

    assert repo.get_history(sid, max_turns=0) == []


def test_message_validation(tmp_path: Path) -> None:
    repo = SQLiteSessionRepository(tmp_path / "s.sqlite")
    sid = repo.create_session()["id"]
    with pytest.raises(ValueError):
        repo.append_message(sid, "tool", "x")
    with pytest.raises(ValueError):
        repo.append_message(sid, "user", "   ")
    with pytest.raises(SessionNotFoundError):
        repo.append_message("missing", "user", "x")


def test_list_limit_and_order(tmp_path: Path) -> None:
    repo = SQLiteSessionRepository(tmp_path / "s.sqlite")
    first = repo.create_session("a")
    repo.create_session("b")
    repo.append_message(first["id"], "user", "bump")
    sessions = repo.list_sessions(limit=1)
    assert len(sessions) == 1
    assert isinstance(repo.list_sessions(limit=0), list)  # 下限钳制到 1


def test_build_history_unit() -> None:
    rows = [
        {"id": 1, "session_id": "s", "role": "user", "content": "abcd", "meta_json": "{}", "created_at": "t1"},
        {
            "id": 2,
            "session_id": "s",
            "role": "assistant",
            "content": json.dumps({"x": 1}),
            "meta_json": "",
            "created_at": "t2",
        },
        {
            "id": 3,
            "session_id": "s",
            "role": "user",
            "content": "efgh",
            "meta_json": '{"ok": true}',
            "created_at": "t3",
        },
    ]
    history = build_history(rows, max_turns=2, max_chars=100)
    assert [item["id"] for item in history] == [1, 2, 3]
    assert history[2]["meta"] == {"ok": True}
    assert build_history(rows, max_turns=0) == []
    truncated = build_history(rows, max_turns=2, max_chars=3)
    assert [item["content"] for item in truncated] == ["efg"]
    assert parse_meta("not json") == {}
    assert parse_meta(None) == {}
    assert parse_meta({"a": 1}) == {"a": 1}


def test_factory(settings: Settings) -> None:
    repo = build_session_repository(settings)
    assert isinstance(repo, SQLiteSessionRepository)
    settings.database_url = "postgres://example/db"
    with pytest.raises(ValueError):
        build_session_repository(settings)
