"""配置测试：默认预算、路径解析、环境变量覆盖与 provider 解析。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagent.config import DEFAULT_LLM_PROVIDERS, Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.agent_max_tool_calls == 4
    assert settings.agent_max_turns == 5
    assert settings.agent_timeout_s == 120.0
    assert settings.database_url == ""
    assert settings.use_mysql is False
    assert settings.code_allowed_roots == [str(Path.home())]
    assert settings.allowed_roots == [Path.home()]
    assert settings.code_store_path == Path(settings.data_root) / "data" / "code_index.sqlite"
    assert settings.session_store_path.name == "sessions.sqlite"
    assert settings.resolved_llm_providers == [dict(item) for item in DEFAULT_LLM_PROVIDERS]


def test_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "idx.sqlite"))
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sess.sqlite"))
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@127.0.0.1:3307/db?charset=utf8mb4")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("AGENT_MAX_TURNS", "4")
    monkeypatch.setenv("AGENT_TIMEOUT_S", "30")
    monkeypatch.setenv("CODE_ALLOWED_ROOTS", json.dumps([str(tmp_path)]))
    settings = Settings(_env_file=None)

    assert settings.agent_max_tool_calls == 3
    assert settings.agent_max_turns == 4
    assert settings.agent_timeout_s == 30.0
    assert settings.use_mysql is True
    assert settings.code_store_path == tmp_path / "idx.sqlite"
    assert settings.session_store_path == tmp_path / "sess.sqlite"
    assert settings.allowed_roots == [tmp_path]


def test_llm_providers_override() -> None:
    payload = [
        {"name": "local", "base_url": "http://127.0.0.1:11434/v1", "model": "qwen"},
        {"name": "second", "base_url": "http://example.invalid/v1", "model": "m"},
    ]
    settings = Settings(llm_providers=json.dumps(payload), _env_file=None)
    assert settings.resolved_llm_providers == payload
    with pytest.raises(TypeError):
        _ = Settings(llm_providers='{"not": "a list"}', _env_file=None).resolved_llm_providers


def test_ensure_dirs(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", _env_file=None)
    settings.ensure_dirs()
    assert (tmp_path / "data" / "data").is_dir()
    assert settings.code_store_path.parent.is_dir()
    assert settings.session_store_path.parent.is_dir()
