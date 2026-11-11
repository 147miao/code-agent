"""codegraph.impact/service 单元测试：图查询映射、深度校验与 SQLite 回退。"""

from __future__ import annotations

import pytest

from codeagent.codegraph import GraphService, GraphUnavailableError, GraphValidationError
from codeagent.codegraph.client import Neo4jClient, Neo4jConfig
from codeagent.codegraph.impact import fallback_impact, graph_impact
from codeagent.config import Settings


class FakeNode(dict):
    def __init__(self, props: dict, labels: list[str]) -> None:
        super().__init__(props)
        self.labels = labels


class FakeRel(dict):
    def __init__(self, type_: str, props: dict, start: FakeNode, end: FakeNode) -> None:
        super().__init__(props)
        self.type = type_
        self.start_node = start
        self.end_node = end


class FakePath:
    def __init__(self, nodes: list[FakeNode], relationships: list[FakeRel]) -> None:
        self.nodes = nodes
        self.relationships = relationships


class FakeClient:
    configured = True

    def __init__(self, records=None, *, fail: bool = False) -> None:
        self._records = records or []
        self.fail = fail

    def records(self, query, parameters=None):
        if self.fail:
            raise GraphUnavailableError("neo4j down")
        return self._records

    def execute(self, query, parameters=None):
        if self.fail:
            raise GraphUnavailableError("neo4j down")
        return []


def _impact_path() -> FakePath:
    source = FakeNode(
        {"repo_id": "repo_x", "kind": "file", "qualname": "pkg/a.py", "path": "pkg/a.py", "name": "a.py"},
        ["File"],
    )
    target = FakeNode(
        {"repo_id": "repo_x", "kind": "file", "qualname": "pkg/b.py", "path": "pkg/b.py", "name": "b.py"},
        ["File"],
    )
    rel = FakeRel(
        "IMPORTS",
        {"repo_id": "repo_x", "edge_key": "IMPORTS:a->b@1", "confidence": 1.0, "lineno": 1, "source": "ast"},
        source,
        target,
    )
    return FakePath([source, target], [rel])


def test_client_connection_failure_does_not_deadlock() -> None:
    def broken_factory(*args, **kwargs):
        raise RuntimeError("connect refused")

    client = Neo4jClient(
        Neo4jConfig("bolt://127.0.0.1:1", "neo4j", "secret", timeout=0.1),
        driver_factory=broken_factory,
    )
    with pytest.raises(GraphUnavailableError):
        _ = client.driver
    assert client._driver is None


def test_graph_impact_maps_nodes_edges_and_imports() -> None:
    client = FakeClient([{"p": _impact_path()}])
    result = graph_impact(client, "repo_x", path="pkg/b.py")
    assert result["source"] == "graph"
    assert result["fallback"] is None
    assert result["imports"] == ["pkg/a.py"]
    assert {(node["label"], node["key"]) for node in result["nodes"]} == {
        ("File", "pkg/a.py"),
        ("File", "pkg/b.py"),
    }
    assert result["edges"][0]["type"] == "IMPORTS"
    assert result["edges"][0]["from"] == "pkg/a.py"


def test_graph_impact_validates_depth_and_query() -> None:
    with pytest.raises(GraphValidationError):
        graph_impact(FakeClient(), "repo_x", path="pkg/b.py", depth=3)
    with pytest.raises(GraphValidationError):
        graph_impact(FakeClient(), "repo_x")


def test_service_impact_falls_back_on_connection_failure(indexed, settings: Settings) -> None:
    service = GraphService(settings, indexed.store, client=FakeClient(fail=True))
    result = service.impact(indexed.repo_id, path="pkg/b.py")
    assert result["fallback"] == "sqlite"
    assert result["source"] == "sqlite"
    assert result["imports"] == ["pkg/a.py"]
    assert result["heuristic"] is True
    assert "neo4j down" in result["graph_error"]


def test_service_impact_falls_back_when_unconfigured(indexed, settings: Settings) -> None:
    unconfigured = Settings(
        data_root=settings.data_root,
        code_allowed_roots=settings.code_allowed_roots,
        neo4j_uri="",
        _env_file=None,
    )
    service = GraphService(unconfigured, indexed.store)
    assert service.configured is False
    result = service.impact(indexed.repo_id, path="pkg/b.py")
    assert result["fallback"] == "sqlite"
    with pytest.raises(GraphUnavailableError):
        service.build(indexed.repo_id)
    with pytest.raises(GraphUnavailableError):
        service.status(indexed.repo_id)


def test_fallback_impact_resolves_qualname(indexed) -> None:
    result = fallback_impact(indexed.store, indexed.repo_id, qualname="ToolExecutor", depth=2)
    assert result["path"] == "pkg/b.py"
    assert result["qualname"] == "ToolExecutor"
    assert result["imports"] == ["pkg/a.py"]
    assert result["depth"] == 2
