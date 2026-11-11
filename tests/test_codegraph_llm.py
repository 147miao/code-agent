"""codegraph.llm_edges 单元测试：候选过滤、输出过滤、缓存、预算与 fanout。"""

from __future__ import annotations

import json

import pytest

from codeagent.codegraph.builder import build_graph_data, build_module_index, merge_llm_edges, symbol_key
from codeagent.codegraph.llm_edges import (
    LlmEdgeConfig,
    build_candidates,
    build_messages,
    extract_llm_edges,
    parse_llm_edges,
)


class FakeChatLLM:
    """最小 chat 替身：记录调用次数，可返回固定 JSON 或抛错。"""

    model = "fake-model"

    def __init__(self, text: str = "", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0
        self.messages: list[list[dict]] = []

    @property
    def clients(self) -> list[object]:
        return [self]

    def chat(self, messages, max_tokens=1024, json_mode=False):
        self.calls += 1
        self.messages.append(messages)
        if self.fail:
            raise RuntimeError("relay down")
        return {"text": self.text, "tokens_in": 10, "tokens_out": 5}


def _edges_payload(target: str = "ToolExecutor", **overrides) -> str:
    edge = {
        "target": target,
        "relation": "CALLS",
        "evidence_line": 5,
        "confidence": 0.9,
    }
    edge.update(overrides)
    return json.dumps({"edges": [edge]})


def test_candidates_include_same_file_and_referenced_imports(indexed) -> None:
    data = indexed.store.export_repo(indexed.repo_id)
    module_index = build_module_index(data["files"])
    candidates = build_candidates("pkg/a.py", data, module_index, LlmEdgeConfig())
    ids = {item["id"] for item in candidates}
    assert {"make", "ToolExecutor"} <= ids
    assert all(item["kind"] != "module" for item in candidates)


def test_candidates_filter_builtins_short_names_and_cap() -> None:
    symbols = [
        {"path": "x.py", "qualname": "print", "kind": "function", "start_line": 1, "end_line": 1},
        {"path": "x.py", "qualname": "ab", "kind": "function", "start_line": 1, "end_line": 1},
        {"path": "x.py", "qualname": "self", "kind": "function", "start_line": 1, "end_line": 1},
        {"path": "x.py", "qualname": "__init__", "kind": "function", "start_line": 1, "end_line": 1},
        {"path": "x.py", "qualname": "valid_name", "kind": "function", "start_line": 1, "end_line": 1},
    ]
    symbols += [
        {
            "path": "x.py",
            "qualname": f"func_{index:02d}",
            "kind": "function",
            "start_line": 10 + index,
            "end_line": 10 + index,
        }
        for index in range(40)
    ]
    data = {"files": [{"path": "x.py"}], "symbols": symbols, "imports": [], "references": [], "chunks": []}
    candidates = build_candidates("x.py", data, {"x": "x.py"}, LlmEdgeConfig())
    ids = {item["id"] for item in candidates}
    assert "valid_name" in ids
    assert not ({"print", "ab", "self", "__init__"} & ids)
    assert len(candidates) == 30
    assert candidates[0]["id"] == "valid_name"


def test_parse_filters_candidate_relation_confidence_line_and_duplicates() -> None:
    candidates = [
        {"id": "ToolExecutor", "qualname": "ToolExecutor", "kind": "class", "path": "pkg/b.py", "name": "ToolExecutor"}
    ]
    payload = json.dumps(
        {
            "edges": [
                {"target": "ToolExecutor", "relation": "CALLS", "evidence_line": 5, "confidence": 0.9},
                {"target": "Nope", "relation": "CALLS", "evidence_line": 5, "confidence": 0.9},
                {"target": "ToolExecutor", "relation": "USES", "evidence_line": 5, "confidence": 0.9},
                {"target": "ToolExecutor", "relation": "CALLS", "evidence_line": 6, "confidence": 0.4},
                {"target": "ToolExecutor", "relation": "CALLS", "evidence_line": 5, "confidence": 0.9},
                {"target": "ToolExecutor", "relation": "TESTS", "evidence_line": 0, "confidence": 0.9},
            ]
        }
    )
    edges, rejected = parse_llm_edges(payload, candidates, LlmEdgeConfig(), loc=10)
    assert len(edges) == 1
    assert edges[0]["relation"] == "CALLS" and edges[0]["evidence_line"] == 5
    assert rejected == {
        "not_candidate": 1,
        "bad_relation": 1,
        "low_confidence": 1,
        "bad_line": 1,
        "duplicate": 1,
    }


def test_parse_applies_fanout_cap() -> None:
    candidates = [
        {"id": f"fn_{index}", "qualname": f"fn_{index}", "kind": "function", "path": "x.py", "name": f"fn_{index}"}
        for index in range(5)
    ]
    payload = json.dumps(
        {
            "edges": [
                {"target": f"fn_{index}", "relation": "CALLS", "evidence_line": index + 1, "confidence": 0.9}
                for index in range(5)
            ]
        }
    )
    edges, _ = parse_llm_edges(payload, candidates, LlmEdgeConfig(fanout=2))
    assert len(edges) == 2


def test_extract_caches_by_file_hash(indexed) -> None:
    data = indexed.store.export_repo(indexed.repo_id)
    config = LlmEdgeConfig(only_paths=("pkg/a.py",))
    first_llm = FakeChatLLM(_edges_payload())
    first = extract_llm_edges(indexed.store, indexed.repo_id, first_llm, config=config, data=data)
    assert first_llm.calls == 1
    assert first["stats"]["llm_calls"] == 1 and first["stats"]["cache_hits"] == 0
    assert first["stats"]["edges"] == 1
    assert first["edges"][0]["type"] == "CALLS"

    second_llm = FakeChatLLM(_edges_payload())
    second = extract_llm_edges(indexed.store, indexed.repo_id, second_llm, config=config, data=data)
    assert second_llm.calls == 0
    assert second["stats"]["cache_hits"] == 1
    assert second["stats"]["edges"] == 1
    assert second["cache"]["entries"] >= 1


def test_extract_skips_llm_failure_and_records_reason(indexed) -> None:
    data = indexed.store.export_repo(indexed.repo_id)
    config = LlmEdgeConfig(only_paths=("pkg/b.py",))
    result = extract_llm_edges(indexed.store, indexed.repo_id, FakeChatLLM(fail=True), config=config, data=data)
    assert result["edges"] == []
    assert result["stats"]["llm_calls"] == 1
    assert result["stats"]["skipped"] == 1
    assert any(item["path"] == "pkg/b.py" and item["reason"].startswith("llm_error") for item in result["skipped"])


def test_token_budget_skips_without_calling_llm(indexed) -> None:
    data = indexed.store.export_repo(indexed.repo_id)
    config = LlmEdgeConfig(budget_tokens=1, only_paths=("pkg/a.py",))
    llm = FakeChatLLM(_edges_payload())
    result = extract_llm_edges(indexed.store, indexed.repo_id, llm, config=config, data=data)
    assert llm.calls == 0
    assert result["stats"]["skipped"] == 1
    assert result["skipped"][0]["reason"] == "token_budget"
    assert build_messages("x.py", "code", [], LlmEdgeConfig(budget_tokens=1)) is None


def test_merge_llm_edges_into_graph_data(indexed) -> None:
    repo = indexed.store.get_repo(indexed.repo_id)
    data = indexed.store.export_repo(indexed.repo_id)
    graph = build_graph_data(repo, data)
    result = extract_llm_edges(
        indexed.store,
        indexed.repo_id,
        FakeChatLLM(_edges_payload()),
        config=LlmEdgeConfig(only_paths=("pkg/a.py",)),
        data=data,
    )
    merge_llm_edges(graph, result)
    assert len(graph["edges"]["CALLS"]) == 1
    edge = graph["edges"]["CALLS"][0]
    assert edge["from_qualname"] == symbol_key("pkg/a.py", "<module>")
    assert edge["to_qualname"] == symbol_key("pkg/b.py", "ToolExecutor")
    assert graph["stats"]["llm"]["edges"] == 1


@pytest.mark.parametrize("bad_text", ["not json", '{"edges": "nope"}', '{"edges": []}'])
def test_parse_handles_bad_or_empty_payloads(bad_text: str) -> None:
    candidates = [
        {"id": "ToolExecutor", "qualname": "ToolExecutor", "kind": "class", "path": "pkg/b.py", "name": "ToolExecutor"}
    ]
    if bad_text == '{"edges": []}':
        edges, rejected = parse_llm_edges(bad_text, candidates, LlmEdgeConfig())
        assert edges == [] and rejected["not_candidate"] == 0
    else:
        with pytest.raises(ValueError):
            parse_llm_edges(bad_text, candidates, LlmEdgeConfig())
