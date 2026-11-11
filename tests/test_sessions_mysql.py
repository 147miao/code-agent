"""MySQL 会话仓储：DSN 解析（纯单测）+ 集成测试（缺 DSN 自动 skip）。"""

from __future__ import annotations

import os

import pytest

from codeagent.sessions.mysql import MySQLSessionRepository, parse_mysql_dsn

DSN_ENV_NAMES = ("CODEAGENT_MYSQL_URL", "DATABASE_URL")


def _dsn() -> str:
    for name in DSN_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value.startswith("mysql"):
            return value
    pytest.skip("未设置 MySQL DSN（CODEAGENT_MYSQL_URL / DATABASE_URL），跳过集成测试")


def test_parse_mysql_dsn() -> None:
    params = parse_mysql_dsn("mysql+pymysql://user:p%40ss@127.0.0.1:3307/codeagent?charset=utf8mb4")
    assert params == {
        "host": "127.0.0.1",
        "port": 3307,
        "user": "user",
        "password": "p@ss",
        "database": "codeagent",
        "charset": "utf8mb4",
    }
    defaults = parse_mysql_dsn("mysql://root@db/codeagent")
    assert defaults["port"] == 3306 and defaults["charset"] == "utf8mb4"
    for bad in ("postgres://x/db", "mysql+pymysql://root@host/"):
        with pytest.raises(ValueError):
            parse_mysql_dsn(bad)


@pytest.mark.integration
def test_mysql_contract() -> None:
    try:
        repo = MySQLSessionRepository(_dsn())
    except Exception as exc:  # noqa: BLE001 - MySQL 未启动时跳过
        pytest.skip(f"MySQL 不可用: {type(exc).__name__}: {exc}")

    session = repo.create_session("集成测试")
    assert repo.get_session(session["id"]) is not None
    assert repo.get_history(session["id"]) == []
    assert repo.append_message(session["id"], "user", "你好", meta={"src": "test"})["id"] > 0
    repo.append_message(session["id"], "assistant", "hello")
    history = repo.get_history(session["id"])
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["meta"] == {"src": "test"}
    assert session["id"] in [item["id"] for item in repo.list_sessions(limit=50)]
    assert repo.delete_session(session["id"]) is True
    assert repo.get_session(session["id"]) is None
    assert repo.delete_session(session["id"]) is False
