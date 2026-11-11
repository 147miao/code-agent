"""Neo4j 集成测试：真实建 schema、写图、status、impact、重建幂等。

默认跳过（addopts = -m 'not integration'）。本机 7689 有 compose neo4j 时：
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_codegraph_neo4j.py -q -m integration
也可用 NEO4J_TEST_URI / NEO4J_TEST_USER / NEO4J_TEST_PASSWORD / NEO4J_TEST_DATABASE 覆盖。
"""
from __future__ import annotations

import os

import pytest

from codeagent.codegraph.client import Neo4jClient, Neo4jConfig
from codeagent.codegraph.schema import delete_repo_graph, ensure_schema
from codeagent.codegraph.service import GraphService
from codeagent.config import Settings

pytestmark = pytest.mark.integration

DEFAULT_TEST_URI = "bolt://127.0.0.1:7689"
DEFAULT_TEST_USER = "neo4j"
DEFAULT_TEST_PASSWORD = "codeagent-local-dev"


def _test_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri=os.environ.get("NEO4J_TEST_URI", DEFAULT_TEST_URI),
        user=os.environ.get("NEO4J_TEST_USER", DEFAULT_TEST_USER),
        password=os.environ.get("NEO4J_TEST_PASSWORD", DEFAULT_TEST_PASSWORD),
        database=os.environ.get("NEO4J_TEST_DATABASE", ""),
        timeout=1.0,
    )


@pytest.fixture
def graph_service(indexed):
    config = _test_config()
    client = Neo4jClient(config)
    try:
        client.driver  # noqa: B018 - 触发 verify_connectivity
    except Exception as exc:  # noqa: BLE001 - 缺服务时跳过而非失败
        pytest.skip(f"Neo4j 不可用（{config.uri}）：{type(exc).__name__}: {exc}")
    settings = Settings(
        neo4j_uri=config.uri,
        neo4j_user=config.user,
        neo4j_password=config.password,
        neo4j_database=config.database,
        neo4j_timeout=config.timeout,
        _env_file=None,
    )
    service = GraphService(settings, indexed.store, client=client)
    ensure_schema(client)
    try:
        yield service, indexed
    finally:
        try:
            delete_repo_graph(client, indexed.repo_id)
        finally:
            client.close()


def test_schema_constraints_and_indexes(graph_service) -> None:
    service, _ = graph_service
    client = service.client
    constraints = {row["name"] for row in client.execute("SHOW CONSTRAINTS YIELD name RETURN name")}
    assert {
        "codegraph_file_key",
        "codegraph_module_key",
        "codegraph_symbol_key",
        "codegraph_name_key",
    } <= constraints
    names = {row["name"] for row in client.execute("SHOW INDEXES YIELD name RETURN name")}
    assert {"codegraph_symbol_name", "codegraph_file_path", "codegraph_calls_repo"} <= names


def test_build_status_impact_and_rebuild(graph_service) -> None:
    service, indexed = graph_service
    first = service.build(indexed.repo_id)
    assert first["nodes"]["File"] == 3
    assert first["edges"]["CONTAINS"] == first["nodes"]["Symbol"]
    assert first["total_nodes"] > 0 and first["total_edges"] > 0
    assert first["resolved_imports"] >= 1

    status = service.status(indexed.repo_id)
    assert status["connected"] is True
    assert status["nodes"] == first["nodes"]
    assert status["total_edges"] == first["total_edges"]

    impact = service.impact(indexed.repo_id, path="pkg/b.py")
    assert impact["source"] == "graph"
    assert impact["fallback"] is None
    assert "pkg/a.py" in impact["imports"]
    assert any(edge["type"] == "IMPORTS" for edge in impact["edges"])

    symbol_impact = service.impact(indexed.repo_id, path="pkg/b.py", qualname="ToolExecutor", depth=2)
    assert symbol_impact["source"] == "graph"
    assert symbol_impact["depth"] == 2

    second = service.build(indexed.repo_id)
    assert second["nodes"] == first["nodes"]
    assert second["edges"] == first["edges"]
