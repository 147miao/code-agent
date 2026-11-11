"""手写 Agent 循环（不依赖 LangGraph）：规划 → 工具 → 编号 contexts → 合成 → 引用校验。

预算：工具调用 ≤ agent_max_tool_calls（默认 6）、LLM 轮数 ≤ agent_max_turns（默认 8）、
总耗时 ≤ agent_timeout_s（默认 120s）。任一预算耗尽后仍会用已收集的 contexts 合成答案。

引用契约：
- 合成 prompt 只允许本轮工具返回的 [n] contexts；
- 校验 path 在索引内、行号 ≤ EOF、编号在范围内；无效引用剔除；
- 检索非空但无有效引用时，用 top2 抽取式兜底；检索为空时拒答。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..citations import parse_structured_answer, verify_code_citations
from ..config import Settings
from . import prompts
from .tools import AgentToolError

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 6
DEFAULT_MAX_TURNS = 8
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_CONTEXTS = 40
DEFAULT_MAX_CONTEXT_CHARS = 1800
DEFAULT_MAX_TOKENS = 1024
MAX_EXTRACTIVE_CONTEXTS = 2
CITE_RE = re.compile(r"\[(\d{1,2})\]")

__all__ = ["AgentLimits", "CodeAgent", "ToolsetProtocol"]


class ToolsetProtocol(Protocol):
    """Agent 需要的工具集契约（AgentTools 或测试替身）。"""

    root: Path
    repo_name: str

    def tool_schemas(self) -> list[dict[str, Any]]:
        ...

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        ...

    def validated_evidence(
        self,
        *,
        path: str,
        start_line: Any,
        end_line: Any,
        qualname: str = "",
        text: str = "",
    ) -> dict[str, Any] | None:
        ...


@dataclass(slots=True)
class AgentLimits:
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_turns: int = DEFAULT_MAX_TURNS
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_contexts: int = DEFAULT_MAX_CONTEXTS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentLimits:
        return cls(
            max_tool_calls=max(1, int(settings.agent_max_tool_calls)),
            max_turns=max(1, int(settings.agent_max_turns)),
            timeout_s=max(0.001, float(settings.agent_timeout_s)),
            max_contexts=DEFAULT_MAX_CONTEXTS,
            max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
            max_tokens=DEFAULT_MAX_TOKENS,
        )


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_lines(path: str) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _strip_invalid_citations(text: str, invalid: Sequence[int]) -> str:
    if not text or not invalid:
        return text or ""
    invalid_set = set(int(item) for item in invalid)

    def replace(match: re.Match[str]) -> str:
        return "" if int(match.group(1)) in invalid_set else match.group(0)

    return re.sub(CITE_RE, replace, text).strip()


class CodeAgent:
    """手写 while 循环的代码库问答 Agent。"""

    def __init__(
        self,
        tools: ToolsetProtocol,
        llm: Any,
        limits: AgentLimits | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tools = tools
        self.llm = llm
        self.limits = limits or AgentLimits()
        self._clock = clock
        self._file_exists, self._read_lines = self._citation_checkers()

    # ---------- public ----------
    def ask(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        text = str(question or "").strip()
        if not text:
            raise ValueError("question 不能为空")
        started = self._clock()
        repo_label = f"{getattr(self.tools, 'repo_name', '') or ''}（{getattr(self.tools, 'root', '')}）"
        messages = prompts.build_planner_messages(
            text,
            history,
            repo_label=repo_label,
            max_tool_calls=self.limits.max_tool_calls,
            max_turns=self.limits.max_turns,
        )
        contexts: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int, str]] = set()
        tool_log: list[dict[str, Any]] = []
        turns = 0
        tokens_in = 0
        tokens_out = 0
        budget_exhausted = False
        timed_out = False
        contexts_truncated = False
        llm_error: str | None = None

        while turns < self.limits.max_turns:
            if self._elapsed(started) >= self.limits.timeout_s:
                timed_out = True
                break
            try:
                response = self.llm.chat_tools(
                    messages,
                    self.tools.tool_schemas(),
                    max_tokens=self.limits.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - LLM 失败降级为拒答/兜底
                llm_error = f"{type(exc).__name__}: {exc}"
                logger.warning("agent planner LLM failed: %s", exc)
                break
            turns += 1
            tokens_in += _to_int((response or {}).get("tokens_in"))
            tokens_out += _to_int((response or {}).get("tokens_out"))
            calls = list((response or {}).get("tool_calls") or [])
            messages.append(
                {
                    "role": "assistant",
                    "content": str((response or {}).get("text") or ""),
                    "tool_calls": calls,
                }
            )
            if not calls:
                break
            halted = False
            for index, call in enumerate(calls):
                if len(tool_log) >= self.limits.max_tool_calls:
                    budget_exhausted = True
                    halted = True
                    break
                call_id, name, arguments, parse_error = self._parse_tool_call(call, index)
                if parse_error:
                    result: dict[str, Any] = {"error": {"code": "invalid_arguments", "message": parse_error}}
                    ok, error, elapsed_ms = False, parse_error, 0
                else:
                    result, ok, error, elapsed_ms = self._run_tool(name, arguments)
                tool_log.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "ok": ok,
                        "error": error,
                        "ms": elapsed_ms,
                        "tool_call_id": call_id,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if ok:
                    for item in self._collect_contexts(name, result):
                        key = (
                            str(item.get("path") or ""),
                            _to_int(item.get("start_line")),
                            _to_int(item.get("end_line")),
                            str(item.get("qualname") or ""),
                        )
                        if key in seen:
                            continue
                        if len(contexts) >= self.limits.max_contexts:
                            contexts_truncated = True
                            break
                        seen.add(key)
                        contexts.append({**item, "n": len(contexts) + 1})
            if halted:
                break

        elapsed = self._elapsed(started)
        if elapsed >= self.limits.timeout_s:
            timed_out = True
        result = self._synthesize(
            text,
            history,
            contexts,
            llm_error=llm_error,
        )
        result["stats"].update(
            {
                "turns": turns,
                "tool_calls": len(tool_log),
                "tokens_in": tokens_in + int(result["stats"].get("tokens_in", 0)),
                "tokens_out": tokens_out + int(result["stats"].get("tokens_out", 0)),
                "latency_ms": int(round(elapsed * 1000)),
                "budget_exhausted": budget_exhausted,
                "timed_out": timed_out,
                "contexts_truncated": contexts_truncated,
                "provider": getattr(self.llm, "last_provider", None),
            }
        )
        result["tool_calls"] = tool_log
        return result

    # ---------- internals ----------
    def _elapsed(self, started: float) -> float:
        return max(0.0, float(self._clock() - started))

    def _citation_checkers(self) -> tuple[Callable[[str], bool], Callable[[str], int]]:
        file_exists = getattr(self.tools, "evidence_file_ok", None)
        read_lines = getattr(self.tools, "line_count", None)
        return (
            file_exists if callable(file_exists) else lambda path: Path(path).is_file(),
            read_lines if callable(read_lines) else _count_lines,
        )

    @staticmethod
    def _parse_tool_call(call: Any, index: int) -> tuple[str, str, dict[str, Any], str | None]:
        if not isinstance(call, dict):
            return f"call_{index}", "", {}, "tool_call 不是对象"
        call_id = str(call.get("id") or f"call_{index}")
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or call.get("name") or "")
        raw = function.get("arguments", call.get("arguments", {}))
        if raw is None:
            return call_id, name, {}, None
        if isinstance(raw, dict):
            return call_id, name, raw, None
        if isinstance(raw, str):
            if not raw.strip():
                return call_id, name, {}, None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return call_id, name, {}, f"arguments 不是合法 JSON: {exc}"
            if isinstance(parsed, dict):
                return call_id, name, parsed, None
            return call_id, name, {}, "arguments 必须是 JSON 对象"
        return call_id, name, {}, "arguments 类型不支持"

    def _run_tool(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool, str | None, int]:
        started = self._clock()
        try:
            result = self.tools.call(name, arguments)
            ok = True
            error = None
        except AgentToolError as exc:
            result = {"error": {"code": exc.code, "message": exc.message}}
            ok, error = False, exc.message
        except Exception as exc:  # noqa: BLE001 - 工具异常转成结构化结果
            logger.warning("agent tool %s failed: %s", name, exc, exc_info=True)
            result = {"error": {"code": "tool_failed", "message": f"{type(exc).__name__}: {exc}"}}
            ok, error = False, str(exc)
        elapsed_ms = int(round(max(0.0, float(self._clock() - started)) * 1000))
        return result, ok, error, elapsed_ms

    def _collect_contexts(self, name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(result, dict) or result.get("error"):
            return []
        candidates: list[dict[str, Any]] = []
        if name == "search_code":
            for hit in result.get("hits") or []:
                candidates.append(
                    {
                        "path": str(hit.get("path") or ""),
                        "start_line": hit.get("start_line"),
                        "end_line": hit.get("end_line"),
                        "qualname": str(hit.get("qualname") or ""),
                        "text": str(hit.get("text") or ""),
                    }
                )
        elif name == "get_symbol":
            if result.get("found"):
                candidates.append(
                    {
                        "path": str(result.get("path") or ""),
                        "start_line": result.get("start_line"),
                        "end_line": result.get("end_line"),
                        "qualname": str(result.get("qualname") or ""),
                        "text": str(result.get("code") or ""),
                    }
                )
        elif name == "read_file":
            text = "\n".join(str(row.get("text") or "") for row in result.get("lines") or [])
            candidates.append(
                {
                    "path": str(result.get("path") or ""),
                    "start_line": result.get("start_line"),
                    "end_line": result.get("end_line"),
                    "qualname": "",
                    "text": text,
                }
            )
        elif name == "grep":
            for match in result.get("matches") or []:
                candidates.append(
                    {
                        "path": str(match.get("path") or ""),
                        "start_line": match.get("line"),
                        "end_line": match.get("line"),
                        "qualname": "",
                        "text": str(match.get("text") or ""),
                    }
                )
        validated: list[dict[str, Any]] = []
        for candidate in candidates:
            item = self.tools.validated_evidence(
                path=candidate["path"],
                start_line=candidate["start_line"],
                end_line=candidate["end_line"],
                qualname=candidate["qualname"],
                text=candidate["text"],
            )
            if not item:
                continue
            item["text"] = str(item.get("text") or "")[: self.limits.max_context_chars]
            if "abs_path" not in item:
                root = getattr(self.tools, "root", None)
                if root is not None:
                    item["abs_path"] = str(Path(root) / str(item.get("path") or ""))
            validated.append(item)
        return validated

    def _public_contexts(self, contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "n": int(context.get("n") or 0),
                "path": str(context.get("path") or ""),
                "start_line": _to_int(context.get("start_line"), 1),
                "end_line": _to_int(context.get("end_line"), 1),
                "qualname": str(context.get("qualname") or ""),
                "text": str(context.get("text") or ""),
            }
            for context in contexts
        ]

    def _refusal(self, question: str, reason: str) -> dict[str, Any]:
        lead, body = prompts.refusal_text(question)
        return {
            "answer": {"lead": lead, "body": body},
            "citations": [],
            "refusal": {"code": "no_evidence", "message": reason},
            "check": "0/0 通过（无引用）",
            "invalid_citations": [],
        }

    def _extractive(self, question: str, contexts: Sequence[dict[str, Any]]) -> dict[str, Any]:
        top = list(contexts)[:MAX_EXTRACTIVE_CONTEXTS]
        lead = prompts.extractive_lead(question, top[0])
        body = prompts.extractive_body(top)
        text = f"{lead}\n\n{body}".strip()
        citations, invalid, check = verify_code_citations(
            text,
            list(contexts),
            file_exists=self._file_exists,
            read_lines=self._read_lines,
        )
        return {
            "answer": {"lead": lead, "body": body},
            "citations": citations,
            "refusal": None,
            "check": check,
            "invalid_citations": invalid,
        }

    def _synthesize(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None,
        contexts: Sequence[dict[str, Any]],
        *,
        llm_error: str | None,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "tokens_in": 0,
            "tokens_out": 0,
            "fallback": None,
            "degraded": False,
            "llm_error": llm_error,
            "contexts": len(contexts),
        }
        if not contexts:
            base = self._refusal(question, llm_error or "本轮没有检索到任何上下文")
            if llm_error:
                base["refusal"] = {
                    "code": "llm_error",
                    "message": f"LLM 调用失败且没有检索到证据: {llm_error}",
                }
            stats["degraded"] = bool(llm_error)
            base["contexts"] = []
            base["stats"] = stats
            base["degraded"] = bool(llm_error)
            return base

        messages = prompts.build_synthesize_messages(question, history, contexts)
        raw = ""
        degraded = bool(llm_error)
        try:
            response = self.llm.chat(messages, max_tokens=self.limits.max_tokens)
            raw = str((response or {}).get("text") or "")
            stats["tokens_in"] = _to_int((response or {}).get("tokens_in"))
            stats["tokens_out"] = _to_int((response or {}).get("tokens_out"))
        except Exception as exc:  # noqa: BLE001 - 合成失败走抽取式兜底
            logger.warning("agent synthesize LLM failed: %s", exc)
            degraded = True
            stats["llm_error"] = f"{type(exc).__name__}: {exc}"

        lead, body = parse_structured_answer(raw) if raw.strip() else ("", "")
        if not lead:
            fallback = self._extractive(question, contexts)
            fallback["contexts"] = self._public_contexts(contexts)
            fallback["stats"] = {**stats, "fallback": "extractive", "degraded": True}
            fallback["degraded"] = True
            return fallback

        answer_text = f"{lead}\n\n{body}".strip()
        citations, invalid, check = verify_code_citations(
            answer_text,
            list(contexts),
            file_exists=self._file_exists,
            read_lines=self._read_lines,
        )
        if invalid:
            lead = _strip_invalid_citations(lead, invalid)
            body = _strip_invalid_citations(body, invalid)
        if not citations:
            fallback = self._extractive(question, contexts)
            fallback["contexts"] = self._public_contexts(contexts)
            fallback["stats"] = {
                **stats,
                "fallback": "extractive",
                "degraded": degraded or bool(invalid),
                "invalid_citations": invalid,
            }
            fallback["invalid_citations"] = invalid
            fallback["degraded"] = fallback["stats"]["degraded"]
            return fallback

        return {
            "answer": {"lead": lead, "body": body},
            "citations": citations,
            "contexts": self._public_contexts(contexts),
            "refusal": None,
            "check": check,
            "invalid_citations": invalid,
            "degraded": degraded,
            "stats": {**stats, "fallback": None},
        }
