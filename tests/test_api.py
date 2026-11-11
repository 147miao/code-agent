"""API 测试（hermetic）：TestClient + 临时 SQLite + fake LLM。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_repo
from fakes import ScriptedLLM, tool_call
from fastapi.testclient import TestClient

from codeagent.api.main import create_app
from codeagent.api.services import AppServices, build_agent_factory
from codeagent.config import Settings
from codeagent.index.indexer import CodeIndexer
from codeagent.index.store import CodeStore
from codeagent.retrieval.service import CodeService
from codeagent.sessions.repository import SQLiteSessionRepository


@pytest.fixture
def api(tmp_path: Path, indexed, settings: Settings, sessions: SQLiteSessionRepository):
    llm = ScriptedLLM(
        tool_responses=[
            {"text": "", "tool_calls": [tool_call("search_code", {"query": "ToolExecutor"})]},
            {"text": "资料已足够", "tool_calls": []},
            {"text": "", "tool_calls": [tool_call("search_code", {"query": "ToolExecutor"})]},
            {"text": "资料已足够", "tool_calls": []},
        ],
        answer="ToolExecutor 定义在 pkg/b.py [1]\n\n它是工具执行类 [1]。",
    )
    services = AppServices(
        settings=settings,
        store=indexed.store,
        indexer=indexed.indexer,
        service=indexed.service,
        sessions=sessions,
        llm=llm,
        agent_factory=build_agent_factory(settings, llm),
    )
    app = create_app(settings, services=services)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            indexed=indexed,
            llm=llm,
            settings=settings,
            sessions=sessions,
            repo_id=indexed.repo_id,
            root=indexed.root,
        )


def test_repos_list_create_delete(api) -> None:
    listed = api.client.get("/api/repos")
    assert listed.status_code == 200
    assert [repo["repo_id"] for repo in listed.json()["repos"]] == [api.repo_id]

    created = api.client.post("/api/repos", json={"root": str(api.root), "name": "renamed"})
    assert created.status_code == 201
    body = created.json()
    assert body["repo"]["name"] == "renamed"
    assert body["report"]["repo_id"] == api.repo_id
    assert body["report"]["files"] >= 2

    deleted = api.client.delete(f"/api/repos/{api.repo_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "repo_id": api.repo_id}
    assert api.client.get("/api/repos").json()["repos"] == []

    missing = api.client.delete(f"/api/repos/{api.repo_id}")
    assert missing.status_code == 404
    error = missing.json()["error"]
    assert error["code"] == "repo_not_found" and error["message"] and error["request_id"]


def test_repos_validation(api) -> None:
    assert api.client.post("/api/repos", json={"root": str(api.root), "unexpected": 1}).status_code == 422
    assert api.client.post("/api/repos", json={"root": ""}).status_code == 422
    assert api.client.post("/api/repos", json={}).status_code == 422


def test_repos_rejects_outside_root_and_missing(tmp_path: Path, sessions: SQLiteSessionRepository) -> None:
    root = make_repo(tmp_path / "inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    db = tmp_path / "idx.sqlite"
    indexer = CodeIndexer(db, [str(root)])
    store = CodeStore(db)
    settings = Settings(
        data_root=tmp_path / "data",
        code_allowed_roots=[str(root)],
        _env_file=None,
    )
    services = AppServices(
        settings=settings,
        store=store,
        indexer=indexer,
        service=CodeService(store),
        sessions=sessions,
        llm=ScriptedLLM(),
    )
    app = create_app(settings, services=services)
    with TestClient(app) as client:
        forbidden = client.post("/api/repos", json={"root": str(outside)})
        assert forbidden.status_code == 403
        assert forbidden.json()["error"]["code"] == "root_not_allowed"

        invalid = client.post("/api/repos", json={"root": str(tmp_path / "missing")})
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_root"


def test_search_symbol_impact_overview(api) -> None:
    hits = api.client.get("/api/search", params={"q": "ToolExecutor", "k": 3})
    assert hits.status_code == 200
    payload = hits.json()
    assert payload["query"] == "ToolExecutor"
    assert payload["count"] >= 1
    assert payload["hits"][0]["path"] == "pkg/b.py"

    assert api.client.get("/api/search", params={}).status_code == 422
    missing_repo = api.client.get("/api/search", params={"q": "x", "repo_id": "missing"})
    assert missing_repo.status_code == 404
    assert missing_repo.json()["error"]["code"] == "repo_not_found"

    symbol = api.client.get("/api/symbol", params={"qualname": "ToolExecutor.run", "repo_id": api.repo_id})
    assert symbol.status_code == 200
    assert symbol.json()["symbol"]["path"] == "pkg/b.py"
    unknown = api.client.get("/api/symbol", params={"qualname": "Nope", "repo_id": api.repo_id})
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "symbol_not_found"

    impact = api.client.get("/api/impact", params={"repo_id": api.repo_id, "path": "pkg/b.py"})
    assert impact.status_code == 200
    assert impact.json()["impact"]["imports"] == ["pkg/a.py"]
    assert impact.json()["impact"]["heuristic"] is True

    overview = api.client.get("/api/overview", params={"repo_id": api.repo_id})
    assert overview.status_code == 200
    assert overview.json()["overview"]["file_count"] == 3


def test_sessions_crud(api) -> None:
    created = api.client.post("/api/sessions", json={"title": "会话 A"})
    assert created.status_code == 201
    session = created.json()["session"]
    assert session["title"] == "会话 A" and session["id"].startswith("sess_")

    listed = api.client.get("/api/sessions")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["sessions"]] == [session["id"]]

    assert api.client.post("/api/sessions", json={"title": "x", "extra": 1}).status_code == 422
    assert api.client.get("/api/sessions", params={"limit": 0}).status_code == 422


def test_ask_with_session_persists_and_reuses_history(api) -> None:
    sid = api.client.post("/api/sessions", json={"title": "ask"}).json()["session"]["id"]
    request = {"question": "ToolExecutor 定义在哪？", "repo_id": api.repo_id, "session_id": sid}
    response = api.client.post("/api/ask", json=request)
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == sid
    assert body["repo_id"] == api.repo_id
    assert body["citations"] and body["citations"][0]["path"] == "pkg/b.py"
    assert body["answer"]["lead"]
    assert body["check"] == "1/1 通过"
    assert body["stats"]["tool_calls"] == 1

    history = api.sessions.get_history(sid)
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["content"] == request["question"]
    assert history[1]["meta"]["citations"][0]["path"] == "pkg/b.py"

    second = api.client.post("/api/ask", json={"question": "继续，它做什么？", "session_id": sid})
    assert second.status_code == 200
    assert len(api.sessions.get_history(sid)) == 4
    planner_messages = api.llm.messages_seen[3][1]
    contents = [message.get("content") for message in planner_messages]
    assert request["question"] in contents


def test_ask_errors(api) -> None:
    missing_session = api.client.post(
        "/api/ask", json={"question": "q", "session_id": "sess_missing"}
    )
    assert missing_session.status_code == 404
    assert missing_session.json()["error"]["code"] == "session_not_found"

    missing_repo = api.client.post("/api/ask", json={"question": "q", "repo_id": "repo_missing"})
    assert missing_repo.status_code == 404
    assert missing_repo.json()["error"]["code"] == "repo_not_found"

    assert api.client.post("/api/ask", json={"question": ""}).status_code == 422
    assert api.client.post("/api/ask", json={"question": "q", "extra": 1}).status_code == 422

    api.client.delete(f"/api/repos/{api.repo_id}")
    no_repo = api.client.post("/api/ask", json={"question": "q"})
    assert no_repo.status_code == 400
    assert no_repo.json()["error"]["code"] == "no_repo_indexed"


def test_reindex_and_health(api) -> None:
    reindexed = api.client.post(f"/api/repos/{api.repo_id}/reindex")
    assert reindexed.status_code == 200
    assert reindexed.json()["report"]["repo_id"] == api.repo_id
    assert api.client.post("/api/repos/repo_missing/reindex").status_code == 404

    health = api.client.get("/api/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert all(payload["deps"].values())


def test_request_id_header(api) -> None:
    response = api.client.get("/api/repos")
    assert response.headers["X-Request-ID"]
    assert len(response.headers["X-Request-ID"]) == 12


def test_dependency_unavailable_returns_503(settings: Settings) -> None:
    services = AppServices(
        settings=settings,
        errors={"service": "store down", "sessions": "sqlite down", "llm": "no providers"},
    )
    app = create_app(settings, services=services)
    with TestClient(app) as client:
        search = client.get("/api/search", params={"q": "x"})
        assert search.status_code == 503
        assert search.json()["error"]["code"] == "service_unavailable"
        assert "store down" in search.json()["error"]["message"]

        sessions = client.post("/api/sessions", json={"title": "x"})
        assert sessions.status_code == 503

        ask = client.post("/api/ask", json={"question": "q"})
        assert ask.status_code == 503

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
