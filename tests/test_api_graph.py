"""API 图端点与 impact 回退测试（fake graph，不连 Neo4j）。"""

from __future__ import annotations

from fakes import ScriptedLLM
from fastapi.testclient import TestClient

from codeagent.api.main import create_app
from codeagent.api.services import AppServices
from codeagent.codegraph import GraphUnavailableError
from codeagent.config import Settings


class FakeGraph:
    configured = True

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.build_calls: list[tuple[str, bool]] = []
        self.impact_calls: list[dict] = []
        self.deleted: list[str] = []

    def describe(self) -> dict:
        return {"configured": self.configured, "uri": "bolt://fake:7689"}

    def build(self, repo_id: str, with_llm: bool = False, **kwargs):
        if self.fail:
            raise GraphUnavailableError("neo4j down")
        self.build_calls.append((repo_id, with_llm))
        return {
            "repo_id": repo_id,
            "nodes": {"File": 3, "Symbol": 12, "Module": 2, "Name": 1},
            "edges": {"CONTAINS": 12, "IMPORTS": 2, "REFERENCES": 9, "CALLS": 0, "TESTS": 0},
            "total_nodes": 18,
            "total_edges": 23,
            "llm": {"edges": 0} if with_llm else None,
        }

    def impact(self, repo_id: str, *, path=None, qualname=None, depth=1):
        self.impact_calls.append({"path": path, "qualname": qualname, "depth": depth})
        return {
            "repo_id": repo_id,
            "path": path or "",
            "qualname": qualname or "",
            "depth": depth,
            "imports": ["pkg/a.py"],
            "nodes": [{"label": "File", "key": "pkg/a.py"}],
            "edges": [{"type": "IMPORTS"}],
            "seeds": 1,
            "fallback": None,
            "source": "graph",
            "heuristic": False,
            "note": "graph",
        }

    def delete(self, repo_id: str):
        if self.fail:
            raise GraphUnavailableError("neo4j down")
        self.deleted.append(repo_id)
        return {"nodes": {"File": 3}, "edges": {"CONTAINS": 12}}

    def status(self, repo_id=None):
        if self.fail:
            raise GraphUnavailableError("neo4j down")
        return {
            "enabled": True,
            "connected": True,
            "repo_id": repo_id or "",
            "nodes": {"File": 3},
            "edges": {"CONTAINS": 12},
            "total_nodes": 3,
            "total_edges": 12,
            "cache": {"entries": 0, "latest": None},
        }


def _app(indexed, settings: Settings, graph, llm=None):
    services = AppServices(
        settings=settings,
        store=indexed.store,
        indexer=indexed.indexer,
        service=indexed.service,
        graph=graph,
        llm=llm,
    )
    return create_app(settings, services=services)


def _client(indexed, settings: Settings, graph, llm=None):
    return TestClient(_app(indexed, settings, graph, llm))


def test_graph_build_and_status(indexed, settings: Settings) -> None:
    graph = FakeGraph()
    with _client(indexed, settings, graph, llm=ScriptedLLM()) as client:
        built = client.post("/api/graph/build", json={"repo_id": indexed.repo_id, "with_llm": False})
        assert built.status_code == 200
        assert built.json()["build"]["total_nodes"] == 18
        assert graph.build_calls == [(indexed.repo_id, False)]

        with_llm = client.post(
            "/api/graph/build",
            json={"repo_id": indexed.repo_id, "with_llm": True},
        )
        assert with_llm.status_code == 200
        assert graph.build_calls[-1] == (indexed.repo_id, True)

        status = client.get("/api/graph/status", params={"repo_id": indexed.repo_id})
        assert status.status_code == 200
        assert status.json()["status"]["total_nodes"] == 3

        missing = client.post("/api/graph/build", json={"repo_id": "repo_missing"})
        assert missing.status_code == 404

        bad = client.post("/api/graph/build", json={"repo_id": indexed.repo_id, "extra": 1})
        assert bad.status_code == 422


def test_graph_build_requires_llm_when_with_llm(indexed, settings: Settings) -> None:
    graph = FakeGraph()
    with _client(indexed, settings, graph, llm=None) as client:
        response = client.post(
            "/api/graph/build",
            json={"repo_id": indexed.repo_id, "with_llm": True},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"
        assert graph.build_calls == []


def test_graph_endpoints_503_when_unconfigured(indexed, settings: Settings) -> None:
    graph = FakeGraph()
    graph.configured = False
    with _client(indexed, settings, graph) as client:
        build = client.post("/api/graph/build", json={"repo_id": indexed.repo_id})
        assert build.status_code == 503
        assert build.json()["error"]["code"] == "graph_unavailable"

        status = client.get("/api/graph/status")
        assert status.status_code == 503

        # impact 不返回 503，而是回退 SQLite。
        fallback = client.get("/api/impact", params={"repo_id": indexed.repo_id, "path": "pkg/b.py"})
        assert fallback.status_code == 200
        assert fallback.json()["impact"]["fallback"] == "sqlite"
        assert fallback.json()["impact"]["imports"] == ["pkg/a.py"]


def test_graph_build_failure_maps_503(indexed, settings: Settings) -> None:
    graph = FakeGraph(fail=True)
    with _client(indexed, settings, graph) as client:
        response = client.post("/api/graph/build", json={"repo_id": indexed.repo_id})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "graph_unavailable"
        status = client.get("/api/graph/status")
        assert status.status_code == 503


def test_impact_prefers_graph_and_has_fallback_field(indexed, settings: Settings) -> None:
    graph = FakeGraph()
    with _client(indexed, settings, graph) as client:
        response = client.get("/api/impact", params={"repo_id": indexed.repo_id, "path": "pkg/b.py"})
        assert response.status_code == 200
        impact = response.json()["impact"]
        assert "fallback" in impact
        assert impact["fallback"] is None
        assert impact["source"] == "graph"
        assert impact["imports"] == ["pkg/a.py"]
        assert graph.impact_calls == [{"path": "pkg/b.py", "qualname": None, "depth": 1}]

        by_qualname = client.get(
            "/api/impact",
            params={"repo_id": indexed.repo_id, "qualname": "ToolExecutor", "depth": 2},
        )
        assert by_qualname.status_code == 200
        assert graph.impact_calls[-1] == {"path": None, "qualname": "ToolExecutor", "depth": 2}


def test_impact_without_path_or_qualname_is_400(indexed, settings: Settings) -> None:
    with _client(indexed, settings, FakeGraph()) as client:
        response = client.get("/api/impact", params={"repo_id": indexed.repo_id})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_argument"


def test_impact_sqlite_fallback_resolves_qualname(indexed, settings: Settings) -> None:
    services = AppServices(settings=settings, store=indexed.store, service=indexed.service)
    app = create_app(settings, services=services)
    with TestClient(app) as client:
        response = client.get(
            "/api/impact",
            params={"repo_id": indexed.repo_id, "qualname": "ToolExecutor"},
        )
        assert response.status_code == 200
        impact = response.json()["impact"]
        assert impact["fallback"] == "sqlite"
        assert impact["path"] == "pkg/b.py"
        assert impact["imports"] == ["pkg/a.py"]
        assert impact["depth"] == 1

        missing = client.get(
            "/api/impact",
            params={"repo_id": indexed.repo_id, "qualname": "NoSuchSymbol"},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "symbol_not_found"


def test_repo_delete_and_reindex_clear_graph(indexed, settings: Settings) -> None:
    graph = FakeGraph()
    with _client(indexed, settings, graph) as client:
        reindexed = client.post(f"/api/repos/{indexed.repo_id}/reindex")
        assert reindexed.status_code == 200
        assert reindexed.json()["graph_cleared"]["edges"]["CONTAINS"] == 12
        assert graph.deleted == [indexed.repo_id]

        deleted = client.delete(f"/api/repos/{indexed.repo_id}")
        assert deleted.status_code == 200
        assert deleted.json()["graph_deleted"]["nodes"]["File"] == 3
        assert graph.deleted == [indexed.repo_id, indexed.repo_id]


def test_health_reports_graph_without_connecting(indexed, settings: Settings) -> None:
    graph = FakeGraph()
    with _client(indexed, settings, graph, llm=ScriptedLLM()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["graph"]["configured"] is True
        assert health.json()["deps"]["graph"] is True
