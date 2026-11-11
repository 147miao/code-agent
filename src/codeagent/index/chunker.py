"""符号到检索块的切分。

规则：
- 每个 class/function/method 符号生成至少一个 chunk；若整个文件没有任何可切符号，
  则退回一个 module 级 chunk（避免只有常量的 __init__.py 无法检索）；
- 单个符号源码 >1500 字或 >120 行时按行切分；每个 part 的上限同时满足两个阈值，
  超长单行按 1500 字硬切，part 保留原行号；
- 切分后 qualname 追加 "#partN"（N 从 1 开始）；单 part 不加后缀；
- chunk_id = sha1(repo_id|path|qualname|start_line|end_line)，确定性且包含 part 信息；
- text 即源码片段本身（保留原始代码，不做摘要/改写）。
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .parser import MODULE_QUALNAME

MAX_CHUNK_CHARS = 1500
MAX_CHUNK_LINES = 120
PART_SEPARATOR = "#part"
CHUNKABLE_KINDS = frozenset({"class", "function", "method"})


def _value(symbol: Any, name: str, default: Any = None) -> Any:
    if isinstance(symbol, Mapping):
        return symbol.get(name, default)
    return getattr(symbol, name, default)


def make_chunk_id(repo_id: str, path: str, qualname: str, start_line: int, end_line: int) -> str:
    """按冻结契约生成确定性 chunk_id。"""
    payload = f"{repo_id}|{path}|{qualname}|{int(start_line)}|{int(end_line)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_chunks(repo_id: str, path: str, symbols: Sequence[Any]) -> list[dict[str, Any]]:
    """把符号列表切为 chunk 字典列表（可直接交给 CodeStore.insert_chunks）。"""
    chunkable = [
        symbol
        for symbol in symbols
        if _value(symbol, "kind") in CHUNKABLE_KINDS and str(_value(symbol, "code") or "").strip()
    ]
    if not chunkable:
        module_symbols = [
            symbol
            for symbol in symbols
            if _value(symbol, "kind") == "module" and str(_value(symbol, "code") or "").strip()
        ]
        chunkable = module_symbols[:1]

    chunks: list[dict[str, Any]] = []
    for symbol in chunkable:
        chunks.extend(_chunk_symbol(repo_id, path, symbol))
    return chunks


def _chunk_symbol(repo_id: str, path: str, symbol: Any) -> list[dict[str, Any]]:
    code = str(_value(symbol, "code") or "")
    if not code.strip():
        return []
    qualname = str(_value(symbol, "qualname") or MODULE_QUALNAME)
    kind = str(_value(symbol, "kind") or "function")
    base_line = int(_value(symbol, "start_line") or 1)
    pieces = _split_lines(code.split("\n"))
    multiple = len(pieces) > 1
    chunks: list[dict[str, Any]] = []
    for part, (start_offset, end_offset, text) in enumerate(pieces, start=1):
        part_qualname = f"{qualname}{PART_SEPARATOR}{part}" if multiple else qualname
        start_line = base_line + start_offset
        end_line = base_line + end_offset
        chunks.append(
            {
                "repo_id": repo_id,
                "chunk_id": make_chunk_id(repo_id, path, part_qualname, start_line, end_line),
                "path": path,
                "qualname": part_qualname,
                "kind": kind,
                "start_line": start_line,
                "end_line": end_line,
                "text": text,
                "part": part,
            }
        )
    return chunks


def _split_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """按 (起始行偏移, 结束行偏移, 文本) 切分；偏移相对符号起始行（0-based）。"""
    if not lines:
        return []
    pieces: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = 0
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if current:
            pieces.append((current_start, current_start + len(current) - 1, "\n".join(current)))
            current = []
            current_chars = 0

    for index, line in enumerate(lines):
        if len(line) > MAX_CHUNK_CHARS:
            flush()
            for offset in range(0, len(line), MAX_CHUNK_CHARS):
                pieces.append((index, index, line[offset : offset + MAX_CHUNK_CHARS]))
            continue
        extra = len(line) + (1 if current else 0)
        if current and (len(current) >= MAX_CHUNK_LINES or current_chars + extra > MAX_CHUNK_CHARS):
            flush()
            extra = len(line)
        if not current:
            current_start = index
        current.append(line)
        current_chars += extra
    flush()
    return pieces
