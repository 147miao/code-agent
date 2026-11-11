"""AgentTools 测试：路径安全、行列上限、检索/符号/影响面与 dispatch。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_index, write_text

from codeagent.agent.tools import AgentPathError, AgentToolError, AgentTools


def make_tools(indexed) -> AgentTools:
    return AgentTools(indexed.service, indexed.root)


def test_discovers_repo(indexed) -> None:
    tools = make_tools(indexed)
    assert tools.repo_id == indexed.repo_id
    assert tools.repo_name == "demo"
    with pytest.raises(AgentToolError) as excinfo:
        AgentTools(indexed.service, indexed.root.parent)
    assert excinfo.value.code == "repo_not_found"


def test_search_code_returns_evidence(indexed) -> None:
    result = make_tools(indexed).search_code("ToolExecutor", k=3)
    assert result["tool"] == "search_code"
    assert result["count"] >= 1
    hit = result["hits"][0]
    assert hit["path"] == "pkg/b.py"
    assert hit["start_line"] >= 1 and hit["end_line"] >= hit["start_line"]
    assert hit["qualname"].startswith("ToolExecutor")
    assert "ToolExecutor" in hit["text"]


def test_get_symbol_and_impact_overview(indexed) -> None:
    tools = make_tools(indexed)
    symbol = tools.get_symbol("ToolExecutor.run")
    assert symbol["found"] is True
    assert symbol["path"] == "pkg/b.py"
    assert symbol["start_line"] >= 1
    assert tools.get_symbol("Missing")["found"] is False

    impact = tools.impact("pkg/b.py")
    assert impact["path"] == "pkg/b.py"
    assert impact["imports"] == ["pkg/a.py"]

    overview = tools.overview()
    assert overview["file_count"] == 3
    assert overview["tool"] == "overview"


def test_read_file_window_and_cap(indexed, sample_repo: Path) -> None:
    write_text(sample_repo / "big.txt", "\n".join(f"line {i}" for i in range(1, 251)) + "\n")
    tools = make_tools(indexed)

    first = tools.read_file("big.txt")
    assert first["total_lines"] == 250
    assert len(first["lines"]) == 200
    assert first["start_line"] == 1 and first["end_line"] == 200
    assert first["truncated"] is True

    window = tools.read_file("big.txt", start=240, end=260)
    assert window["start_line"] == 240
    assert window["end_line"] == 250
    assert len(window["lines"]) == 11
    assert window["truncated"] is False

    with pytest.raises(AgentToolError) as excinfo:
        tools.read_file("big.txt", start=300)
    assert excinfo.value.code == "range_out_of_bounds"
    with pytest.raises(AgentToolError):
        tools.read_file("big.txt", start=10, end=5)


def test_read_file_absolute_path_inside_root(indexed, sample_repo: Path) -> None:
    tools = make_tools(indexed)
    absolute = str(sample_repo / "pkg" / "b.py")
    result = tools.read_file(absolute)
    assert result["path"] == "pkg/b.py"
    assert any("class ToolExecutor" in row["text"] for row in result["lines"])


def test_list_dir_cap(indexed, sample_repo: Path) -> None:
    many = sample_repo / "many"
    many.mkdir()
    for index in range(205):
        write_text(many / f"f{index:03d}.py", "x = 1\n")
    tools = make_tools(indexed)
    result = tools.list_dir("many")
    assert result["count"] == 200
    assert result["truncated"] is True
    assert all(entry["kind"] == "file" for entry in result["entries"])

    with pytest.raises(AgentToolError):
        tools.list_dir("missing-dir")


def test_grep_cap_prefix_and_invalid_regex(tmp_path: Path) -> None:
    root = tmp_path / "grepo"
    write_text(root / "hits.py", "\n".join("HIT = 1" for _ in range(150)) + "\n")
    write_text(root / "other.py", "nothing here\n")
    indexed = build_index(root, tmp_path)
    tools = AgentTools(indexed.service, root)

    result = tools.grep("HIT")
    assert result["count"] == 100
    assert result["truncated"] is True
    assert all(match["path"] == "hits.py" for match in result["matches"])
    assert result["matches"][0]["line"] == 1

    prefixed = tools.grep("HIT", path_prefix="other.py")
    assert prefixed["matches"] == []
    with pytest.raises(AgentToolError) as excinfo:
        tools.grep("(")
    assert excinfo.value.code == "invalid_pattern"
    with pytest.raises(AgentToolError):
        tools.grep("HIT", path_prefix="no-such-dir")


def test_path_escape_is_rejected(tmp_path: Path, indexed, sample_repo: Path) -> None:
    outside = write_text(tmp_path / "outside.py", "SECRET = 1\n")
    link = sample_repo / "link.py"
    link.symlink_to(outside)
    tools = make_tools(indexed)

    for bad in ("../outside.py", str(outside), "link.py"):
        with pytest.raises(AgentPathError) as excinfo:
            tools.read_file(bad)
        assert excinfo.value.code == "path_not_allowed"
    with pytest.raises(AgentPathError):
        tools.list_dir("../")
    with pytest.raises(AgentPathError):
        tools.grep("SECRET", path_prefix="..")
    with pytest.raises(AgentPathError):
        tools.call("read_file", {"path": "../outside.py"})


def test_symlink_inside_root_is_rejected(indexed, sample_repo: Path) -> None:
    alias = sample_repo / "alias.py"
    alias.symlink_to(sample_repo / "pkg" / "b.py")
    with pytest.raises(AgentPathError):
        make_tools(indexed).read_file("alias.py")


def test_validated_evidence_filters(indexed, tmp_path: Path) -> None:
    write_text(indexed.root / "notes.txt", "not indexed\n")
    tools = make_tools(indexed)
    assert tools.validated_evidence(path="pkg/b.py", start_line=1, end_line=2) is not None
    assert tools.validated_evidence(path="notes.txt", start_line=1, end_line=1) is None
    assert tools.validated_evidence(path="../outside.py", start_line=1, end_line=1) is None
    assert tools.validated_evidence(path="pkg/b.py", start_line=999, end_line=1000) is None
    clamped = tools.validated_evidence(path="pkg/b.py", start_line=1, end_line=9999)
    assert clamped is not None and clamped["end_line"] <= 20


def test_call_dispatch_errors(indexed) -> None:
    tools = make_tools(indexed)
    with pytest.raises(AgentToolError) as excinfo:
        tools.call("rm_rf", {})
    assert excinfo.value.code == "unknown_tool"
    with pytest.raises(AgentToolError) as excinfo:
        tools.call("read_file", ["not", "a", "dict"])
    assert excinfo.value.code == "invalid_args"
    with pytest.raises(AgentToolError) as excinfo:
        tools.call("read_file", {"path": "pkg/b.py", "unknown": 1})
    assert excinfo.value.code == "invalid_args"


def test_tool_schemas_are_defensive_copies() -> None:
    tools = AgentTools.__new__(AgentTools)
    schemas = tools.tool_schemas()
    names = {item["function"]["name"] for item in schemas}
    assert {"search_code", "get_symbol", "read_file", "list_dir", "grep", "impact", "overview"} <= names
    schemas[0]["function"]["name"] = "mutated"
    assert tools.tool_schemas()[0]["function"]["name"] == "search_code"
    json.dumps(schemas)  # schema 必须可序列化
