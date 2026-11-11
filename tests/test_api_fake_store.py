"""API 装配与 CodeStore 解耦验证：fake store 也能驱动只读端点。"""

from __future__ import annotations

from pathlib import Path

from fakes import FakeStore, ScriptedLLM
from fastapi.testclient import TestClient

from codeagent.api.main import create_app
from codeagent.api.services import AppServices
from codeagent.config import Settings
from codeagent.retrieval.service import CodeService
from codeagent.sessions.repository import SQLiteSessionRepository


def test_api_with_fake_store(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", code_allowed_roots=[str(tmp_path)], _env_file=None)
    store = FakeStore()
    services = AppServices(
        settings=settings,
        store=store,
        indexer=None,
        service=CodeService(store),
        sessions=SQLiteSessionRepository(tmp_path / "sessions.sqlite"),
        llm=ScriptedLLM(),
    )
    app = create_app(settings, services=services)
    with TestClient(app) as client:
        repos = client.get("/api/repos").json()["repos"]
        assert repos[0]["repo_id"] == "repo_fake"

        hits = client.get("/api/search", params={"q": "ToolExecutor"})
        assert hits.status_code == 200
        assert hits.json()["count"] == 1

        symbol = client.get("/api/symbol", params={"qualname": "ToolExecutor", "repo_id": "repo_fake"})
        assert symbol.status_code == 200
        assert symbol.json()["symbol"]["path"] == "pkg/b.py"

        overview = client.get("/api/overview", params={"repo_id": "repo_fake"})
        assert overview.json()["overview"]["file_count"] == 1

        # indexer 为 None：写端点 503，但只读端点仍可用
        assert client.post("/api/repos", json={"root": "/tmp"}).status_code == 503
