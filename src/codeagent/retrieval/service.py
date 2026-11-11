"""代码库检索 / 符号 / 概览 / 影响面服务。

本模块只依赖 plan-codeagent.md 冻结的 CodeStore 契约（用 typing.Protocol 描述），
不 import 索引实现；测试注入 fake 或真实 CodeStore 均可运行。

范围说明：
- Python-only，检索走 SQLite 双 FTS5（unicode61 + trigram，RRF 合并）；
- impact 仅文件级反向 import，启发式，不承诺精确调用链；
- ask（LLM 问答）不在本模块，交给 Agent 工作流；引用校验在 codeagent.citations。
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_K = 8
MAX_SEARCH_K = 50
IMPACT_NOTE = "文件级反向 import（启发式，非精确调用链）"

__all__ = [
    "DEFAULT_SEARCH_K",
    "IMPACT_NOTE",
    "MAX_SEARCH_K",
    "CodeRepoNotFound",
    "CodeService",
    "CodeServiceError",
    "CodeStoreProtocol",
    "CodeValidationError",
]


class CodeStoreProtocol(Protocol):
    """CodeStore 的冻结契约（本模块只依赖这些方法）。"""

    def add_repo(self, root: str, name: str) -> dict[str, Any]:
        ...

    def list_repos(self) -> list[dict[str, Any]]:
        ...

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        ...

    def delete_repo(self, repo_id: str) -> bool:
        ...

    def search(
        self,
        query: str,
        k: int = 8,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        ...

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        ...

    def overview(self, repo_id: str) -> dict[str, Any]:
        ...


class CodeServiceError(RuntimeError):
    """代码服务相关错误的基类。"""


class CodeRepoNotFound(CodeServiceError):
    """repo_id 不存在。"""


class CodeValidationError(CodeServiceError, ValueError):
    """调用参数不合法。"""


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_repo(raw: dict[str, Any] | None) -> dict[str, Any]:
    """把 CodeStore 返回的 repo dict 归一化为 API 冻结字段。"""
    data = raw or {}
    return {
        "repo_id": _to_text(data.get("repo_id") or data.get("id")),
        "root": _to_text(data.get("root")),
        "name": _to_text(data.get("name")),
        "created_at": _to_text(data.get("created_at")),
        "updated_at": _to_text(data.get("updated_at")),
        "file_count": _to_int(data.get("file_count")),
        "symbol_count": _to_int(data.get("symbol_count")),
        "status": _to_text(data.get("status")) or "ready",
    }


def _normalize_hit(raw: dict[str, Any], repo_id: str | None = None) -> dict[str, Any]:
    """归一化检索命中；兼容 chunk 的 text 字段与旧 code 字段。"""
    start_line = _to_int(raw.get("start_line"))
    end_line = _to_int(raw.get("end_line"), start_line)
    start_line = max(start_line, 1)
    end_line = max(end_line, start_line)
    return {
        "repo_id": _to_text(raw.get("repo_id") or repo_id),
        "chunk_id": _to_text(raw.get("chunk_id")),
        "path": _to_text(raw.get("path")),
        "qualname": _to_text(raw.get("qualname")),
        "kind": _to_text(raw.get("kind")),
        "start_line": start_line,
        "end_line": end_line,
        "score": _to_float(raw.get("score")),
        "text": _to_text(raw.get("text") or raw.get("code")),
    }


def _normalize_symbol(
    raw: dict[str, Any], repo_id: str, qualname: str, path: str | None = None
) -> dict[str, Any]:
    start_line = _to_int(raw.get("start_line"))
    end_line = _to_int(raw.get("end_line"), start_line)
    start_line = max(start_line, 1)
    end_line = max(end_line, start_line)
    return {
        "repo_id": _to_text(raw.get("repo_id") or repo_id),
        "path": _to_text(raw.get("path") or path),
        "qualname": _to_text(raw.get("qualname") or qualname),
        "kind": _to_text(raw.get("kind")),
        "start_line": start_line,
        "end_line": end_line,
        "docstring": _to_text(raw.get("docstring")),
        "code": _to_text(raw.get("code") or raw.get("text")),
    }


def _overview_metric(raw: dict[str, Any], keys: Sequence[str], default: int = 0) -> int:
    """overview 的计数键兼容：file_count / files / top_files 等。"""
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)
        return _to_int(value, default)
    return default


def _top_paths(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    paths: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            text = _to_text(item.get("path") or item.get("name"))
            symbols = _to_int(item.get("symbols") or item.get("symbol_count") or item.get("count"))
        else:
            text = _to_text(item)
            symbols = 0
        if text:
            paths.append({"path": text, "symbols": symbols})
    return paths[:10]


def _kind_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _to_int(count) for key, count in value.items() if _to_int(count) >= 0}


def _normalize_overview(repo_id: str, repo: dict[str, Any], raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw or {}
    return {
        "repo_id": repo_id,
        "name": _to_text(data.get("name") or repo.get("name")),
        "root": _to_text(data.get("root") or repo.get("root")),
        "file_count": _overview_metric(
            data, ("file_count", "files", "top_files"), _to_int(repo.get("file_count"))
        ),
        "symbol_count": _overview_metric(
            data, ("symbol_count", "symbols"), _to_int(repo.get("symbol_count"))
        ),
        "kinds": _kind_counts(data.get("kinds") or data.get("kind_counts")),
        "top_paths": _top_paths(data.get("top_paths") or data.get("top_files") or data.get("paths")),
        "status": _to_text(data.get("status") or repo.get("status")) or "ready",
        "note": _to_text(data.get("note")),
    }


class CodeService:
    """代码检索 / 符号 / 概览 / 影响面的门面。"""

    def __init__(self, store: CodeStoreProtocol) -> None:
        self.store = store

    # ---- repos ----
    def list_repos(self) -> list[dict[str, Any]]:
        return [_normalize_repo(item) for item in (self.store.list_repos() or []) if isinstance(item, dict)]

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        repo_id = _to_text(repo_id).strip()
        if not repo_id:
            return None
        raw = self.store.get_repo(repo_id)
        return _normalize_repo(raw) if raw else None

    def delete_repo(self, repo_id: str) -> bool:
        repo_id = _to_text(repo_id).strip()
        if not repo_id:
            return False
        return bool(self.store.delete_repo(repo_id))

    def _require_repo(self, repo_id: str) -> dict[str, Any]:
        repo = self.get_repo(repo_id)
        if repo is None:
            raise CodeRepoNotFound(f"repo not found: {repo_id}")
        return repo

    # ---- search / symbol / overview ----
    def search_code(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        query = _to_text(query).strip()
        if not query:
            raise CodeValidationError("query must not be empty")
        k = max(1, min(_to_int(k, DEFAULT_SEARCH_K), MAX_SEARCH_K))
        repo_id = _to_text(repo_id).strip() or None
        if repo_id is not None:
            self._require_repo(repo_id)
        path_prefix = _to_text(path_prefix).strip() or None
        kind = _to_text(kind).strip().lower() or None
        raw_hits = self.store.search(
            query,
            k=k,
            repo_id=repo_id,
            path_prefix=path_prefix,
            kind=kind,
        )
        return [_normalize_hit(item, repo_id) for item in (raw_hits or []) if isinstance(item, dict)]

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        repo_id = _to_text(repo_id).strip()
        qualname = _to_text(qualname).strip()
        path = _to_text(path).strip() or None
        if not repo_id or not qualname:
            raise CodeValidationError("repo_id and qualname must not be empty")
        self._require_repo(repo_id)
        raw = self.store.get_symbol(repo_id, qualname, path=path)
        if not raw:
            return None
        return _normalize_symbol(raw, repo_id, qualname, path)

    def overview(self, repo_id: str) -> dict[str, Any]:
        repo_id = _to_text(repo_id).strip()
        if not repo_id:
            raise CodeValidationError("repo_id must not be empty")
        repo = self._require_repo(repo_id)
        raw = self.store.overview(repo_id) or {}
        if not isinstance(raw, dict):
            raw = {}
        return _normalize_overview(repo_id, repo, raw)

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        repo_id = _to_text(repo_id).strip()
        path = _to_text(path).strip()
        if not repo_id or not path:
            raise CodeValidationError("repo_id and path must not be empty")
        self._require_repo(repo_id)
        raw = self.store.reverse_imports(repo_id, path) or []
        return [_to_text(item) for item in raw if _to_text(item)]

    def impact(self, repo_id: str, path: str) -> dict[str, Any]:
        repo_id = _to_text(repo_id).strip()
        path = _to_text(path).strip()
        imports = self.reverse_imports(repo_id, path)
        return {
            "repo_id": repo_id,
            "path": path,
            "imports": imports,
            "heuristic": True,
            "note": IMPACT_NOTE,
        }
