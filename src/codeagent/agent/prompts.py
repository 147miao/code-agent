"""Agent 提示词与上下文渲染。

- 规划阶段：只负责调用工具收集证据，不产出最终答案；
- 合成阶段：只允许使用本轮工具产生的 [n] contexts，强制引用；
- 历史消息截断后拼入两阶段 prompt，避免长会话挤占预算。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 1500
MAX_CONTEXT_CHARS = 1800

PLANNER_SYSTEM = """你是 code-agent 的代码库检索规划器。你的任务不是直接回答问题，而是调用工具收集证据。
规则：
1. 只能使用提供的工具；优先 search_code 定位，再用 get_symbol / read_file / grep 展开。
2. 工具参数里的路径必须是仓库内的相对路径；越界路径会被拒绝。
3. 最多调用 {max_tool_calls} 次工具、最多 {max_turns} 轮；预算耗尽后会强制进入答案合成。
4. 证据足够时，不要再调用工具，直接输出一行“资料已足够”并结束。
5. 不要使用工具之外的知识，不要输出 [n] 引用或最终答案。
6. 历史消息里的 [n] 属于上一轮，不要复用。"""

SYNTH_SYSTEM = """你是代码库问答助手。只能依据【检索上下文】回答，不得使用上下文之外的知识。
规则：
1. 每个关键结论后必须用 [n] 标注来源（n 是上下文编号），可以连续标注多个，例如 [1][3]。
2. 只允许引用列出的编号；禁止编造编号，禁止引用未列出的资料。
3. 上下文不足以回答时，明确说明“检索到的代码不足以回答该问题”，不要猜测。
4. 用与【问题】相同的语言回答（中文问题用中文，英文问题用英文，不要混用）。
5. 输出格式（纯文本）：第一行一句话结论（不超过 60 字，可带 [n]）；空行；之后 2-5 句展开，每句关键结论带 [n]。
6. 不要输出 JSON、Markdown 标题、代码块或“结论：”前缀。
7. 历史消息只用于理解对话；历史里的 [n] 属于上一轮，不要复用，只引用本轮列出的上下文。"""


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def history_messages(
    history: Sequence[dict[str, Any]] | None,
    *,
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[dict[str, str]]:
    """把会话历史裁剪为 OpenAI messages（只保留 user/assistant，取最近若干条）。"""
    items = list(history or [])[-max_messages:]
    messages: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role not in {"user", "assistant"} or not content.strip():
            continue
        messages.append({"role": role, "content": content[:max_chars]})
    return messages


def build_planner_messages(
    question: str,
    history: Sequence[dict[str, Any]] | None,
    *,
    repo_label: str,
    max_tool_calls: int,
    max_turns: int,
) -> list[dict[str, Any]]:
    system = PLANNER_SYSTEM.format(max_tool_calls=max_tool_calls, max_turns=max_turns)
    user = f"仓库：{repo_label}\n问题：{question}\n请开始检索。"
    return [
        {"role": "system", "content": system},
        *history_messages(history),
        {"role": "user", "content": user},
    ]


def _context_block(contexts: Sequence[dict[str, Any]]) -> str:
    lines = ["【检索上下文】"]
    for context in contexts:
        number = int(context.get("n") or 0)
        path = str(context.get("path") or "")
        start = int(context.get("start_line") or 1)
        end = int(context.get("end_line") or start)
        qualname = str(context.get("qualname") or "")
        header = f"[{number}] {path}:{start}-{end}"
        if qualname:
            header += f"（{qualname}）"
        lines.append(header)
        lines.append(str(context.get("text") or "")[:MAX_CONTEXT_CHARS])
        lines.append("")
    return "\n".join(lines)


def build_synthesize_messages(
    question: str,
    history: Sequence[dict[str, Any]] | None,
    contexts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    user = (
        f"{_context_block(contexts)}\n"
        f"【问题】{question}\n"
        "【输出要求】只依据上面列出的上下文回答，用 [n] 标注；第一行一句话结论，空行后展开。"
    )
    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        *history_messages(history),
        {"role": "user", "content": user},
    ]


def refusal_text(question: str) -> tuple[str, str]:
    """检索为空时的拒答文案（语言跟随问题）。"""
    if is_chinese(question):
        return (
            "未能在代码库中检索到与该问题相关的证据。",
            "请提供更具体的符号名、文件路径或关键词，或先确认该仓库已完成索引。",
        )
    return (
        "No relevant evidence was found in the indexed codebase.",
        "Please provide a more specific symbol, path, or keyword, or confirm the repository has been indexed.",
    )


def extractive_lead(question: str, context: dict[str, Any]) -> str:
    path = str(context.get("path") or "")
    start = int(context.get("start_line") or 1)
    end = int(context.get("end_line") or start)
    if is_chinese(question):
        return f"根据代码库检索结果，最相关的是 {path}:{start}-{end} [1]。"
    return f"The most relevant indexed code is {path}:{start}-{end} [1]."


def extractive_body(contexts: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, context in enumerate(contexts, start=1):
        path = str(context.get("path") or "")
        start = int(context.get("start_line") or 1)
        end = int(context.get("end_line") or start)
        qualname = str(context.get("qualname") or "")
        snippet = " ".join(str(context.get("text") or "").split())[:160]
        label = f"[{index}] {path}:{start}-{end}"
        if qualname:
            label += f"（{qualname}）"
        parts.append(f"{label}：{snippet}")
    return "\n".join(parts)
