"""Agent 工具集：全部限定 repo root（realpath），并限制行数/条数。

安全边界：
- 所有路径先展开为 realpath，必须位于 repo root 内；符号链接指向 root 之外会被拒绝；
- read_file 每次最多 200 行、list_dir 最多 200 项、grep 最多 100 条匹配；
- search_code/get_symbol/impact/overview 走 CodeService，路径与行号来自索引；
- 工具错误以 AgentToolError(code, message) 抛出，由 Agent 循环转成工具结果。
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

from ..index.scanner import is_excluded_dir
from ..retrieval.service import CodeService

MAX_READ_LINES = 200
MAX_LIST_ENTRIES = 200
MAX_GREP_MATCHES = 100
MAX_GREP_LINE_CHARS = 400
MAX_FILE_BYTES = 1_000_000
MAX_CONTEXT_TEXT = 1800
MAX_SEARCH_K = 20
DEFAULT_SEARCH_K = 8

TOOL_NAMES = ("search_code", "get_symbol", "read_file", "list_dir", "grep", "impact", "overview")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "按关键词/符号名检索代码块（FTS5 双表 RRF），返回 path、行号、qualname 与片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，支持中文/标识符"},
                    "k": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_K, "default": DEFAULT_SEARCH_K},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": "按 qualname（如 Class.method）精确获取符号定义与代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "qualname": {"type": "string"},
                    "path": {"type": "string", "description": "可选：仓库内相对路径"},
                },
                "required": ["qualname"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取仓库内文件的一段，每次最多 200 行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "仓库内相对路径"},
                    "start": {"type": "integer", "minimum": 1, "description": "起始行（1-based）"},
                    "end": {"type": "integer", "minimum": 1, "description": "结束行（含）"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出仓库内目录项，最多 200 项。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "仓库内相对目录，默认仓库根"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "在已索引文件中按正则搜索，最多返回 100 条匹配（path/行号/该行文本）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python 正则"},
                    "path_prefix": {"type": "string", "description": "可选：限定目录/文件前缀"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "impact",
            "description": "文件级反向 import（启发式）：哪些文件 import 了该文件。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "仓库内相对路径"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "overview",
            "description": "仓库概览：文件/符号数、符号类型分布、最大的文件。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class AgentToolError(ValueError):
    """Agent 工具错误；code 用于结构化返回。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AgentPathError(AgentToolError):
    """路径越界或符号链接逃逸。"""

    def __init__(self, message: str) -> None:
        super().__init__("path_not_allowed", message)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class AgentTools:
    """面向单个仓库的只读工具集。"""

    def __init__(
        self,
        service: CodeService,
        root: str | Path,
        repo_id: str | None = None,
        *,
        max_read_lines: int = MAX_READ_LINES,
        max_list_entries: int = MAX_LIST_ENTRIES,
        max_grep_matches: int = MAX_GREP_MATCHES,
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self.service = service
        self.root = Path(root).expanduser().resolve()
        self.max_read_lines = max(1, int(max_read_lines))
        self.max_list_entries = max(1, int(max_list_entries))
        self.max_grep_matches = max(1, int(max_grep_matches))
        self.max_file_bytes = max(1, int(max_file_bytes))
        self._line_counts: dict[str, int] = {}
        self._indexed: set[str] = set()
        self._indexed_loaded = False
        if repo_id:
            self.repo_id = str(repo_id)
            repo = self.service.get_repo(self.repo_id) or {}
            self.repo_name = str(repo.get("name") or self.root.name)
        else:
            self.repo_id, self.repo_name = self._discover_repo()

    # ---------- repo / path ----------
    def _discover_repo(self) -> tuple[str, str]:
        for repo in self.service.list_repos():
            try:
                repo_root = Path(str(repo.get("root") or "")).expanduser().resolve()
            except OSError:
                continue
            if repo_root == self.root:
                return str(repo.get("repo_id") or ""), str(repo.get("name") or "") or self.root.name
        raise AgentToolError("repo_not_found", f"仓库未索引: {self.root}")

    def _resolve(self, path: str | Path, *, must_exist: bool = True) -> tuple[str, Path]:
        raw = str(path or "").strip().replace("\\", "/")
        if raw in ("", "."):
            if must_exist and not self.root.is_dir():
                raise AgentToolError("not_found", f"仓库根不存在: {self.root}")
            return "", self.root
        candidate = Path(raw)
        lexical = candidate if candidate.is_absolute() else self.root / candidate
        try:
            abs_path = lexical.resolve(strict=False)
        except OSError as exc:  # pragma: no cover - 极端文件系统错误
            raise AgentPathError(f"无法解析路径: {raw}") from exc
        if not abs_path.is_relative_to(self.root):
            raise AgentPathError(f"路径越界（必须在仓库根内）: {raw}")
        if lexical.is_symlink():
            raise AgentPathError(f"拒绝符号链接: {raw}")
        rel = abs_path.relative_to(self.root).as_posix()
        if rel == ".":
            rel = ""
        if must_exist and not abs_path.exists():
            raise AgentToolError("not_found", f"路径不存在: {rel or '.'}")
        return rel, abs_path

    def _file(self, path: str | Path) -> tuple[str, Path]:
        rel, abs_path = self._resolve(path)
        if not abs_path.is_file():
            raise AgentToolError("not_a_file", f"不是文件: {rel or '.'}")
        return rel, abs_path

    def _path_indexed(self, rel_path: str) -> bool:
        if not self._indexed_loaded:
            store = getattr(self.service, "store", None)
            lister = getattr(store, "list_files", None)
            if callable(lister):
                try:
                    self._indexed = {str(item.get("path") or "") for item in lister(self.repo_id)}
                    self._indexed_loaded = True
                except Exception:  # noqa: BLE001 - 索引不可用时退回文件系统校验
                    self._indexed_loaded = False
        if self._indexed_loaded:
            return rel_path in self._indexed
        return (self.root / rel_path).is_file()

    def _text_files(self) -> list[str]:
        if not self._indexed_loaded:
            self._path_indexed("__probe__")
        if self._indexed_loaded:
            return sorted(self._indexed)
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not is_excluded_dir(name) and not os.path.islink(os.path.join(dirpath, name))
            )
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                full = Path(dirpath) / name
                if full.is_symlink():
                    continue
                found.append(full.relative_to(self.root).as_posix())
        return found

    # ---------- citation helpers ----------
    def line_count(self, abs_path: str | Path) -> int:
        key = str(abs_path)
        if key not in self._line_counts:
            with Path(abs_path).open("r", encoding="utf-8", errors="replace") as handle:
                self._line_counts[key] = sum(1 for _ in handle)
        return self._line_counts[key]

    def evidence_file_ok(self, abs_path: str | Path) -> bool:
        """引用校验用：文件存在、位于 root 内且属于索引。"""
        try:
            rel = Path(abs_path).resolve(strict=False).relative_to(self.root).as_posix()
        except (OSError, ValueError):
            return False
        return bool(rel) and self._path_indexed(rel) and (self.root / rel).is_file()

    def validated_evidence(
        self,
        *,
        path: str,
        start_line: Any,
        end_line: Any,
        qualname: str = "",
        text: str = "",
    ) -> dict[str, Any] | None:
        """把工具结果转成可引用 context；路径越界/不在索引/行号非法时返回 None。"""
        try:
            rel, abs_path = self._resolve(path)
        except AgentToolError:
            return None
        if not rel or not abs_path.is_file() or not self._path_indexed(rel):
            return None
        start = max(1, _to_int(start_line, 1))
        end = max(start, _to_int(end_line, start))
        try:
            total = self.line_count(abs_path)
        except OSError:
            return None
        if start > total:
            return None
        end = min(end, total)
        return {
            "path": rel,
            "abs_path": str(abs_path),
            "start_line": start,
            "end_line": end,
            "qualname": str(qualname or ""),
            "text": str(text or "")[:MAX_CONTEXT_TEXT],
        }

    # ---------- tools ----------
    def search_code(self, query: str, k: int = DEFAULT_SEARCH_K) -> dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            raise AgentToolError("invalid_args", "query 不能为空")
        limit = max(1, min(_to_int(k, DEFAULT_SEARCH_K), MAX_SEARCH_K))
        tokens = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)))[:8]
        queries = [text]
        if tokens:
            token_query = " ".join(tokens)
            if token_query.lower() != text.lower():
                queries.append(token_query)
        merged: dict[str, dict[str, object]] = {}
        for qi, query in enumerate(queries):
            for rank, hit in enumerate(self.service.search_code(query, k=limit * 2, repo_id=self.repo_id)):
                key = str(hit.get("chunk_id") or f"{hit.get('path')}:{hit.get('start_line')}:{hit.get('end_line')}")
                item = merged.setdefault(key, {**hit, "_rrf": 0.0})
                item["_rrf"] = float(item["_rrf"]) + (1.0 / (60 + rank + 1)) * (1.2 if qi else 1.0)
        hits = sorted(merged.values(), key=lambda item: -float(item.get("_rrf") or 0.0))[:limit]
        items: list[dict[str, Any]] = []
        for hit in hits:
            path = str(hit.get("path") or "")
            try:
                rel, _ = self._resolve(path, must_exist=False)
            except AgentToolError:
                continue
            if not rel:
                continue
            items.append(
                {
                    "path": rel,
                    "start_line": _to_int(hit.get("start_line"), 1),
                    "end_line": _to_int(hit.get("end_line"), _to_int(hit.get("start_line"), 1)),
                    "qualname": str(hit.get("qualname") or ""),
                    "kind": str(hit.get("kind") or ""),
                    "score": round(float(hit.get("score") or 0.0), 6),
                    "text": str(hit.get("text") or ""),
                }
            )
        return {"tool": "search_code", "query": text, "count": len(items), "hits": items}

    def get_symbol(self, qualname: str, path: str | None = None) -> dict[str, Any]:
        name = str(qualname or "").strip()
        if not name:
            raise AgentToolError("invalid_args", "qualname 不能为空")
        rel_path = None
        if path:
            rel_path, _ = self._resolve(path, must_exist=False)
        symbol = self.service.get_symbol(self.repo_id, name, path=rel_path)
        if symbol is None:
            return {"tool": "get_symbol", "found": False, "qualname": name}
        code_lines = str(symbol.get("code") or "").splitlines()
        truncated = len(code_lines) > self.max_read_lines
        return {
            "tool": "get_symbol",
            "found": True,
            "path": str(symbol.get("path") or rel_path or ""),
            "qualname": str(symbol.get("qualname") or name),
            "kind": str(symbol.get("kind") or ""),
            "start_line": _to_int(symbol.get("start_line"), 1),
            "end_line": _to_int(symbol.get("end_line"), 1),
            "docstring": str(symbol.get("docstring") or ""),
            "code": "\n".join(code_lines[: self.max_read_lines]),
            "truncated": truncated,
        }

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> dict[str, Any]:
        rel, abs_path = self._file(path)
        try:
            size = abs_path.stat().st_size
        except OSError as exc:
            raise AgentToolError("unreadable", f"无法读取: {rel}") from exc
        if size > self.max_file_bytes:
            raise AgentToolError("file_too_large", f"文件过大（>{self.max_file_bytes} 字节）: {rel}")
        try:
            with abs_path.open("r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError as exc:
            raise AgentToolError("unreadable", f"无法读取: {rel}") from exc
        total = len(lines)
        if total == 0:
            return {
                "tool": "read_file",
                "path": rel,
                "start_line": 1,
                "end_line": 1,
                "total_lines": 0,
                "lines": [],
                "truncated": False,
            }
        start_line = max(1, _to_int(start, 1))
        if start_line > total:
            raise AgentToolError("range_out_of_bounds", f"起始行超过文件总行数 {total}: {rel}")
        requested_end = _to_int(end, start_line + self.max_read_lines - 1)
        if requested_end < start_line:
            raise AgentToolError("invalid_args", "end 不能小于 start")
        window_end = min(requested_end, total, start_line + self.max_read_lines - 1)
        rows = [
            {"line": lineno, "text": lines[lineno - 1]}
            for lineno in range(start_line, window_end + 1)
        ]
        return {
            "tool": "read_file",
            "path": rel,
            "start_line": start_line,
            "end_line": window_end,
            "total_lines": total,
            "lines": rows,
            "truncated": window_end < total,
        }

    def list_dir(self, path: str = "") -> dict[str, Any]:
        rel, abs_path = self._resolve(path)
        if not abs_path.is_dir():
            raise AgentToolError("not_a_directory", f"不是目录: {rel or '.'}")
        try:
            with os.scandir(abs_path) as scanner:
                scanned = sorted(
                    scanner,
                    key=lambda item: (not item.is_dir(follow_symlinks=False), item.name),
                )
        except OSError as exc:
            raise AgentToolError("unreadable", f"无法读取目录: {rel or '.'}") from exc
        entries: list[dict[str, Any]] = []
        truncated = False
        for item in scanned:
            if item.is_symlink():
                continue
            is_dir = item.is_dir(follow_symlinks=False)
            try:
                size = 0 if is_dir else int(item.stat(follow_symlinks=False).st_size)
            except OSError:
                size = 0
            entries.append({"name": item.name, "kind": "dir" if is_dir else "file", "size": size})
            if len(entries) >= self.max_list_entries:
                truncated = len(scanned) > len(entries)
                break
        return {
            "tool": "list_dir",
            "path": rel,
            "entries": entries,
            "count": len(entries),
            "truncated": truncated,
        }

    def grep(self, pattern: str, path_prefix: str | None = None) -> dict[str, Any]:
        raw_pattern = str(pattern or "")
        if not raw_pattern:
            raise AgentToolError("invalid_args", "pattern 不能为空")
        try:
            regex = re.compile(raw_pattern)
        except re.error as exc:
            raise AgentToolError("invalid_pattern", f"正则无效: {exc}") from exc
        prefix = ""
        if path_prefix:
            prefix, _ = self._resolve(path_prefix)
        candidates = self._text_files()
        if prefix:
            candidates = [item for item in candidates if item == prefix or item.startswith(prefix + "/")]
        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        for rel in candidates:
            if len(matches) >= self.max_grep_matches:
                truncated = True
                break
            abs_path = self.root / rel
            if abs_path.is_symlink() or not abs_path.is_file():
                continue
            try:
                if abs_path.stat().st_size > self.max_file_bytes:
                    continue
                with abs_path.open("r", encoding="utf-8", errors="replace") as handle:
                    lines = handle.read().splitlines()
            except OSError:
                continue
            scanned += 1
            for lineno, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                matches.append({"path": rel, "line": lineno, "text": line[:MAX_GREP_LINE_CHARS]})
                if len(matches) >= self.max_grep_matches:
                    truncated = True
                    break
        return {
            "tool": "grep",
            "pattern": raw_pattern,
            "path_prefix": prefix,
            "matches": matches,
            "count": len(matches),
            "scanned_files": scanned,
            "truncated": truncated,
        }

    def impact(self, path: str) -> dict[str, Any]:
        rel, _ = self._file(path)
        impact = self.service.impact(self.repo_id, rel)
        return {"tool": "impact", **impact}

    def overview(self) -> dict[str, Any]:
        overview = self.service.overview(self.repo_id)
        return {"tool": "overview", **overview}

    # ---------- dispatch ----------
    def tool_schemas(self) -> list[dict[str, Any]]:
        return copy.deepcopy(TOOL_SCHEMAS)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            raise AgentToolError("unknown_tool", f"未知工具: {name}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise AgentToolError("invalid_args", f"{name} 的参数必须是 JSON 对象")
        handler = getattr(self, name)
        try:
            return handler(**arguments)
        except TypeError as exc:
            raise AgentToolError("invalid_args", f"{name}: {exc}") from exc
