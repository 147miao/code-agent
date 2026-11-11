"""引用解析与校验（本地实现，不依赖 rag / store / Milvus）。

来源：从旧 RAG-M 的 rag/citations.py、rag/graph.py 的 parse_structured_answer 与
code/service.py 的 verify_code_citations 抽取，去掉对旧包的全部依赖。

对外契约：
- parse_citations(text)：按出现顺序去重解析 [n]；
- verify_citations(text, n_evidence)：范围校验（通用文档 RAG 语义）；
- parse_structured_answer(text)：拆分“第一行结论 + 空行 + 展开”；
- verify_code_citations(answer, contexts)：代码引用校验，要求 path 非空、行号合法、
  文件存在且 end_line 不超过文件总行数。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CITE_RE = re.compile(r"\[(\d{1,2})\]")
LEAD_PREFIX_RE = re.compile(r"^(结论|答案|一句话结论)[：:]\s*")

__all__ = [
    "parse_citations",
    "parse_structured_answer",
    "verify_citations",
    "verify_code_citations",
]


def parse_citations(text: str) -> list[int]:
    """按出现顺序解析 [n] 引用并去重。"""
    seen: list[int] = []
    for match in CITE_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def verify_citations(text: str, n_evidence: int) -> tuple[list[int], list[int], str]:
    """返回 (有效引用, 无效引用, 校验文案)。"""
    cited = parse_citations(text)
    valid = [number for number in cited if 1 <= number <= n_evidence]
    invalid = [number for number in cited if number < 1 or number > n_evidence]
    if not cited:
        return [], [], "0/0 通过（无引用）"
    return valid, invalid, f"{len(valid)}/{len(cited)} 通过"


def parse_structured_answer(text: str) -> tuple[str, str]:
    """解析“第一行结论 + 空行 + 展开”格式，兼容单行/代码围栏残留。"""
    raw = (text or "").strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + fence + r"[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?" + fence + r"$", "", raw).strip()
    lines = raw.splitlines()
    lead, rest_start = "", 0
    for index, line in enumerate(lines):
        if line.strip():
            lead = line.strip()
            rest_start = index + 1
            break
    lead = LEAD_PREFIX_RE.sub("", lead).strip()
    body = "\n".join(lines[rest_start:]).strip()
    if body:
        return lead, body
    parts = re.split(r"(?<=[。！？.!?])\s*", lead, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return lead, ""


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _citation_numbers(answer: str | Sequence[int]) -> list[int]:
    if isinstance(answer, str):
        return parse_citations(answer)
    if isinstance(answer, Sequence) and not isinstance(answer, (bytes, bytearray)):
        numbers: list[int] = []
        for item in answer:
            number = _to_int(item, 0)
            if number and number not in numbers:
                numbers.append(number)
        return numbers
    return []


def _default_file_exists(path: str) -> bool:
    return Path(path).is_file()


def _default_read_lines(path: str) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def verify_code_citations(
    answer: str | Sequence[int],
    contexts: Sequence[dict[str, Any]],
    *,
    file_exists: Callable[[str], bool] | None = None,
    read_lines: Callable[[str], int] | None = None,
) -> tuple[list[dict[str, Any]], list[int], str]:
    """校验回答中的 [n] 引用。

    params:
        answer: LLM 回答文本（解析 [n]）或已解析的引用编号序列；
        contexts: 每条含 path / abs_path / start_line / end_line / qualname；
        file_exists: 可注入（默认 Path.is_file），参数为解析后的绝对路径；
        read_lines: 可注入（默认读取文件计行数），参数同上。

    returns:
        (有效引用列表, 无效引用编号列表, 校验文案)

    引用有效的条件：编号在范围内、path 非空、start_line >= 1、end_line >= start_line、
    文件存在且 end_line 不超过文件总行数。无效引用不会出现在返回的 citations 中。
    """
    numbers = _citation_numbers(answer)
    exists = file_exists or _default_file_exists
    line_count = read_lines or _default_read_lines
    citations: list[dict[str, Any]] = []
    invalid: list[int] = []

    for number in numbers:
        if number < 1 or number > len(contexts):
            invalid.append(number)
            continue
        context = contexts[number - 1]
        path = _to_text(context.get("path"))
        abs_path = _to_text(context.get("abs_path")) or path
        start_line = _to_int(context.get("start_line"))
        end_line = _to_int(context.get("end_line"), start_line)
        if not path or not abs_path or start_line < 1 or end_line < start_line:
            invalid.append(number)
            continue
        try:
            if not exists(abs_path):
                invalid.append(number)
                continue
            total_lines = _to_int(line_count(abs_path))
        except Exception:
            logger.debug("citation file check failed: %s", abs_path, exc_info=True)
            invalid.append(number)
            continue
        if total_lines < end_line:
            invalid.append(number)
            continue
        citations.append(
            {
                "n": number,
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "qualname": _to_text(context.get("qualname")),
            }
        )

    check = "0/0 通过（无引用）" if not numbers else f"{len(citations)}/{len(numbers)} 通过"
    return citations, invalid, check
