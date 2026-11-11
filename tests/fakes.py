"""测试替身：脚本化 LLM、Agent 工具集、CodeStore。"""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

from codeagent.agent.tools import AgentToolError


def tool_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "call_" + uuid.uuid4().hex[:8],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {}, ensure_ascii=False)},
    }


class ScriptedLLM:
    """按脚本返回 chat_tools 结果；chat 返回固定 answer。

    tool_responses 每项形如 {"text": ..., "tool_calls": [...]}；用尽后返回无 tool_calls。
    """

    def __init__(
        self,
        tool_responses: list[dict[str, Any]] | None = None,
        answer: str = "",
        *,
        fail_tools: bool = False,
        fail_chat: bool = False,
    ) -> None:
        self.tool_responses = list(tool_responses or [])
        self.answer = answer
        self.fail_tools = fail_tools
        self.fail_chat = fail_chat
        self.messages_seen: list[tuple[str, list[dict[str, Any]], Any]] = []
        self.last_provider = "fake"

    @property
    def providers(self) -> list[object]:
        return [object()]

    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        self.messages_seen.append(("tools", copy.deepcopy(messages), copy.deepcopy(tools)))
        if self.fail_tools:
            raise RuntimeError("planner down")
        item = self.tool_responses.pop(0) if self.tool_responses else {"text": "", "tool_calls": []}
        return {
            "text": str(item.get("text") or ""),
            "tool_calls": list(item.get("tool_calls") or []),
            "tokens_in": 1,
            "tokens_out": 1,
        }

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 1024) -> dict[str, Any]:
        self.messages_seen.append(("chat", copy.deepcopy(messages), None))
        if self.fail_chat:
            raise RuntimeError("synth down")
        return {"text": self.answer, "tokens_in": 2, "tokens_out": 3}

    def synth_messages(self) -> list[dict[str, Any]]:
        for kind, messages, _ in self.messages_seen:
            if kind == "chat":
                return messages
        return []


class FakeTools:
    """内存版工具集：files 为 {相对路径: 行列表}，results 为按工具名的固定返回。"""

    def __init__(
        self,
        files: dict[str, list[str]] | None = None,
        results: dict[str, Any] | None = None,
        *,
        root: str = "/fake/repo",
    ) -> None:
        self.files = files or {"pkg/b.py": [f"line {index}" for index in range(1, 31)]}
        self.results = results or {}
        self.root = Path(root)
        self.repo_name = "fake"

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "search_code", "parameters": {}}}]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.results.get(name)
        if result is None:
            raise AgentToolError("unknown_tool", f"未知工具: {name}")
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    def _rel(self, abs_path: str) -> str:
        prefix = str(self.root).rstrip("/") + "/"
        return abs_path[len(prefix):] if abs_path.startswith(prefix) else ""

    def validated_evidence(
        self,
        *,
        path: str,
        start_line: Any,
        end_line: Any,
        qualname: str = "",
        text: str = "",
    ) -> dict[str, Any] | None:
        if not path or path.startswith("/") or ".." in Path(path).parts:
            return None
        if path not in self.files:
            return None
        total = len(self.files[path])
        try:
            start = max(1, int(start_line or 1))
            end = max(start, int(end_line or start))
        except (TypeError, ValueError):
            return None
        if start > total:
            return None
        return {
            "path": path,
            "abs_path": str(self.root / path),
            "start_line": start,
            "end_line": min(end, total),
            "qualname": qualname,
            "text": text,
        }

    def evidence_file_ok(self, abs_path: str) -> bool:
        rel = self._rel(str(abs_path))
        return bool(rel) and rel in self.files

    def line_count(self, abs_path: str) -> int:
        rel = self._rel(str(abs_path))
        if rel not in self.files:
            raise FileNotFoundError(rel)
        return len(self.files[rel])


class FakeStore:
    """CodeStore 契约的最小内存实现（供 API 装配测试使用）。"""

    def __init__(
        self,
        repos: list[dict[str, Any]] | None = None,
        hits: list[dict[str, Any]] | None = None,
        symbol: dict[str, Any] | None = None,
    ) -> None:
        self.repos = list(repos) if repos is not None else [
            {
                "repo_id": "repo_fake",
                "root": "/fake/repo",
                "name": "fake",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "file_count": 1,
                "symbol_count": 1,
                "status": "ready",
            }
        ]
        self.hits = list(hits) if hits is not None else [
            {
                "repo_id": "repo_fake",
                "chunk_id": "chunk-1",
                "path": "pkg/b.py",
                "qualname": "ToolExecutor",
                "kind": "class",
                "start_line": 4,
                "end_line": 9,
                "score": 0.5,
                "text": "class ToolExecutor: ...",
            }
        ]
        self.symbol = symbol or {
            "repo_id": "repo_fake",
            "path": "pkg/b.py",
            "qualname": "ToolExecutor",
            "kind": "class",
            "start_line": 4,
            "end_line": 9,
            "docstring": "",
            "code": "class ToolExecutor: ...",
        }

    def add_repo(self, root: str, name: str) -> dict[str, Any]:
        return self.repos[0]

    def list_repos(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.repos)

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        for repo in self.repos:
            if repo["repo_id"] == repo_id:
                return copy.deepcopy(repo)
        return None

    def delete_repo(self, repo_id: str) -> bool:
        before = len(self.repos)
        self.repos = [repo for repo in self.repos if repo["repo_id"] != repo_id]
        return len(self.repos) != before

    def search(
        self,
        query: str,
        k: int = 8,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        return copy.deepcopy(self.hits[:k])

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        if self.symbol.get("qualname") != qualname:
            return None
        return copy.deepcopy(self.symbol)

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        return ["pkg/a.py"]

    def overview(self, repo_id: str) -> dict[str, Any]:
        return {
            "repo_id": repo_id,
            "name": "fake",
            "root": "/fake/repo",
            "file_count": 1,
            "symbol_count": 1,
            "kinds": {"class": 1},
            "top_paths": [{"path": "pkg/b.py", "symbols": 1}],
            "status": "ready",
        }
