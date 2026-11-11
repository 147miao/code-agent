"""LLM 候选边（CALLS/TESTS）：候选集、JSON 输出过滤、按 file_hash 缓存。

候选集（每文件）：
- 同文件符号（排除 <module>、builtins、self/cls、短名 <3）；
- import 能解析到本仓库文件时，该文件里被当前文件引用到的名字对应的符号；
- 去重后 cap 30（同文件优先）。

调用约定：FallbackLLM.chat(messages, max_tokens, json_mode=True)，只接受候选内的
target、relation ∈ {CALLS, TESTS}、confidence >= 阈值；每个文件最多 fanout 条边。
失败/超预算只跳过该文件并在结果里记录，不中断整次构建。
"""
from __future__ import annotations

import builtins
import contextlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .builder import (
    MODULE_QUALNAME,
    build_module_index,
    normalize_path,
    resolve_import,
    symbol_key,
    symbol_name,
)

BUILTIN_NAMES = frozenset(dir(builtins))
EXCLUDED_NAMES = frozenset({"self", "cls"})
MIN_NAME_CHARS = 3
ALLOWED_RELATIONS = frozenset({"CALLS", "TESTS"})
MAX_EDGES_PER_FILE = 50  # 兜底；实际默认值由 LlmEdgeConfig.fanout 控制
TOKEN_CHARS = 4
MIN_CODE_CHARS = 200
FENCE = chr(96) * 3
CACHE_TABLE = "codegraph_llm_cache"
CACHE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
    repo_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    path TEXT NOT NULL,
    edges_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (repo_id, file_hash, model)
);
CREATE INDEX IF NOT EXISTS idx_codegraph_llm_cache_repo ON {CACHE_TABLE}(repo_id);
"""

SYSTEM_PROMPT = (
    "You are a precise Python code graph extractor. "
    "Given one file and a fixed candidate list, find real call/test relationships from this "
    "file's module to candidate symbols. Only use candidates, never invent targets. "
    "Return a JSON object: {\"edges\": [{\"target\": \"<candidate id>\", "
    "\"relation\": \"CALLS|TESTS\", \"evidence_line\": <int>, "
    "\"confidence\": <0..1>}]}. Use TESTS only when the file clearly tests the target."
)


@dataclass(frozen=True)
class LlmEdgeConfig:
    """LLM 边抽取参数；默认值来自 Settings。"""

    max_candidates: int = 30
    confidence: float = 0.6
    fanout: int = 50
    budget_tokens: int = 12000
    max_output_tokens: int = 4096
    max_files: int | None = None
    only_paths: tuple[str, ...] | None = None

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> LlmEdgeConfig:
        values = {
            "max_candidates": int(getattr(settings, "graph_llm_max_candidates", 30)),
            "confidence": float(getattr(settings, "graph_llm_confidence", 0.6)),
            "fanout": int(getattr(settings, "graph_llm_fanout", 50)),
            "budget_tokens": int(getattr(settings, "graph_llm_budget_tokens", 12000)),
            "max_output_tokens": int(getattr(settings, "graph_llm_max_output_tokens", 4096)),
        }
        values.update(overrides)
        return cls(**values)

    def with_paths(self, paths: Sequence[str] | None) -> LlmEdgeConfig:
        if not paths:
            return self
        return LlmEdgeConfig(**{**self.__dict__, "only_paths": tuple(normalize_path(path) for path in paths)})


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class LlmEdgeCache:
    """按 (repo_id, file_hash, model) 缓存 LLM 成功结果；失败不缓存。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=3)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(CACHE_SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, repo_id: str, file_hash: str, model: str) -> list[dict[str, Any]] | None:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT edges_json FROM {CACHE_TABLE} WHERE repo_id = ? AND file_hash = ? AND model = ?",
                (repo_id, file_hash, model),
            ).fetchone()
        if row is None:
            return None
        try:
            edges = json.loads(str(row["edges_json"]))
        except (json.JSONDecodeError, TypeError):
            return None
        return edges if isinstance(edges, list) else None

    def put(
        self,
        repo_id: str,
        file_hash: str,
        model: str,
        path: str,
        edges: Sequence[Mapping[str, Any]],
    ) -> None:
        payload = json.dumps([dict(edge) for edge in edges], ensure_ascii=False)
        with self._connection() as conn:
            conn.execute(
                f"INSERT INTO {CACHE_TABLE} (repo_id, file_hash, model, path, edges_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(repo_id, file_hash, model) "
                "DO UPDATE SET path = excluded.path, edges_json = excluded.edges_json, "
                "created_at = excluded.created_at",
                (repo_id, file_hash, model, path, payload, _utcnow()),
            )

    def stats(self, repo_id: str | None = None) -> dict[str, Any]:
        where = " WHERE repo_id = ?" if repo_id else ""
        params: tuple[Any, ...] = (repo_id,) if repo_id else ()
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS entries, MAX(created_at) AS latest FROM {CACHE_TABLE}{where}",
                params,
            ).fetchone()
        return {"entries": int(row["entries"] or 0), "latest": row["latest"]}


def model_label(llm: Any) -> str:
    """缓存键里的模型标识：优先 provider 链的 model，测试替身退化为类名。"""
    clients = getattr(llm, "clients", None) or []
    labels = [str(getattr(client, "model", "") or type(client).__name__) for client in clients]
    return ",".join(labels) or type(llm).__name__


def eligible_name(name: str) -> bool:
    """过滤 builtins/self/cls/短名/dunder，减少无意义候选。"""
    return (
        bool(name)
        and len(name) >= MIN_NAME_CHARS
        and name not in BUILTIN_NAMES
        and name not in EXCLUDED_NAMES
        and not name.startswith("__")
    )


def _symbol_row(item: Mapping[str, Any]) -> dict[str, Any]:
    path = normalize_path(str(item.get("path") or ""))
    qualname = str(item.get("qualname") or "")
    return {
        "path": path,
        "qualname": qualname,
        "kind": str(item.get("kind") or ""),
        "name": symbol_name(qualname),
        "start_line": int(item.get("start_line") or 1),
        "end_line": int(item.get("end_line") or 1),
    }


def _candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": symbol_key(str(row["path"]), str(row["qualname"])),
        "path": str(row["path"]),
        "qualname": str(row["qualname"]),
        "kind": str(row["kind"]),
        "name": str(row["name"]),
    }


def _assign_display_ids(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同名候选用 path::qualname 消歧；否则用更短的 qualname，降低模型回显错误。"""
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item["qualname"]] = counts.get(item["qualname"], 0) + 1
    for item in candidates:
        item["id"] = item["qualname"] if counts[item["qualname"]] == 1 else item["id"]
    return candidates


def build_candidates(
    path: str,
    data: Mapping[str, Sequence[Mapping[str, Any]]],
    module_index: Mapping[str, str],
    config: LlmEdgeConfig,
) -> list[dict[str, Any]]:
    """构造单文件候选集；纯函数，便于单测。"""
    current_path = normalize_path(path)
    symbols = [
        _symbol_row(item)
        for item in data.get("symbols") or []
        if item.get("path") and item.get("qualname")
    ]
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in symbols:
        by_path.setdefault(row["path"], []).append(row)
    for rows in by_path.values():
        rows.sort(key=lambda row: (row["start_line"], row["qualname"]))
    current = by_path.get(current_path, [])

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: Mapping[str, Any]) -> None:
        if row["kind"] == "module" or not eligible_name(str(row["name"])):
            return
        item = _candidate(row)
        if item["id"] in seen:
            return
        seen.add(item["id"])
        candidates.append(item)

    for row in current:
        add(row)

    references = {
        str(item.get("name") or "")
        for item in data.get("references") or []
        if normalize_path(str(item.get("src_path") or "")) == current_path
    }
    referenced = {name for name in references if eligible_name(name)}
    imports = [
        item
        for item in data.get("imports") or []
        if normalize_path(str(item.get("src_path") or "")) == current_path
    ]
    for item in imports:
        target = resolve_import(
            current_path,
            str(item.get("module") or ""),
            int(item.get("level") or 0),
            module_index,
        )
        if target is None:
            continue
        for row in by_path.get(target, []):
            if row["name"] in referenced:
                add(row)

    return _assign_display_ids(candidates[: max(0, int(config.max_candidates))])


def _file_code(path: str, data: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    current_path = normalize_path(path)
    chunks = [
        item
        for item in data.get("chunks") or []
        if normalize_path(str(item.get("path") or "")) == current_path
    ]
    for chunk in chunks:
        if str(chunk.get("qualname") or "") == MODULE_QUALNAME and str(chunk.get("kind") or "") == "module":
            return str(chunk.get("text") or "")
    # 没有 module chunk 时按行号拼接，避免重复片段。
    ordered = sorted(chunks, key=lambda item: (int(item.get("start_line") or 0), str(item.get("chunk_id") or "")))
    seen_text: set[str] = set()
    parts: list[str] = []
    for chunk in ordered:
        text = str(chunk.get("text") or "")
        if text and text not in seen_text:
            seen_text.add(text)
            parts.append(text)
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + TOKEN_CHARS - 1) // TOKEN_CHARS)


def build_messages(
    path: str,
    code: str,
    candidates: Sequence[Mapping[str, Any]],
    config: LlmEdgeConfig,
) -> list[dict[str, str]] | None:
    """构造 chat messages；代码超出 token 预算时截断，截断后仍不足则返回 None。"""
    if not code.strip():
        return None
    candidate_lines = [
        f"{index}. target={item['id']} kind={item['kind']} name={item['name']} path={item['path']}"
        for index, item in enumerate(candidates, start=1)
    ]
    header = (
        f"File: {path}\n"
        "Candidates (target id must be copied exactly):\n"
        + "\n".join(candidate_lines)
        + f"\n\nCode:\n{FENCE}python\n"
    )
    footer = f"\n{FENCE}\nReturn JSON only."
    overhead = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(header) + estimate_tokens(footer)
    allowed_chars = max(0, (int(config.budget_tokens) - overhead) * TOKEN_CHARS)
    if allowed_chars < MIN_CODE_CHARS:
        return None
    if len(code) > allowed_chars:
        code = code[:allowed_chars] + "\n# ... truncated by token budget ..."
    user = header + code + footer
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith(FENCE):
        stripped = stripped.strip(chr(96))
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM 输出中没有 JSON 对象")
    return json.loads(stripped[start : end + 1])


def parse_llm_edges(
    text: str,
    candidates: Sequence[Mapping[str, Any]],
    config: LlmEdgeConfig,
    *,
    loc: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """解析并过滤 LLM 输出；返回 (edges, rejects)。非法项只计数，不抛异常。"""
    payload = _extract_json(text)
    items = payload.get("edges") if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise ValueError("LLM 输出缺少 edges 数组")

    by_id = {str(item["id"]): item for item in candidates}
    by_qualname: dict[str, list[Mapping[str, Any]]] = {}
    for item in candidates:
        by_qualname.setdefault(str(item["qualname"]), []).append(item)

    accepted: list[dict[str, Any]] = []
    rejected = {"not_candidate": 0, "bad_relation": 0, "low_confidence": 0, "bad_line": 0, "duplicate": 0}
    seen: set[tuple[str, str, int]] = set()
    cap = min(max(0, int(config.fanout)), MAX_EDGES_PER_FILE)
    for item in items:
        if not isinstance(item, Mapping):
            rejected["not_candidate"] += 1
            continue
        target_id = str(item.get("target") or "").strip()
        candidate = by_id.get(target_id)
        if candidate is None:
            matches = by_qualname.get(target_id) or []
            candidate = matches[0] if len(matches) == 1 else None
        if candidate is None:
            rejected["not_candidate"] += 1
            continue
        relation = str(item.get("relation") or "").strip().upper()
        if relation not in ALLOWED_RELATIONS:
            rejected["bad_relation"] += 1
            continue
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < float(config.confidence):
            rejected["low_confidence"] += 1
            continue
        try:
            evidence_line = int(item.get("evidence_line"))
        except (TypeError, ValueError):
            rejected["bad_line"] += 1
            continue
        if evidence_line < 1:
            rejected["bad_line"] += 1
            continue
        if loc > 0:
            evidence_line = min(evidence_line, loc)
        key = (str(candidate["id"]), relation, evidence_line)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        accepted.append(
            {
                "target_id": str(candidate["id"]),
                "target_qualname": str(candidate["qualname"]),
                "target_kind": str(candidate["kind"]),
                "target_path": str(candidate["path"]),
                "relation": relation,
                "evidence_line": evidence_line,
                "confidence": round(confidence, 3),
            }
        )
        if len(accepted) >= cap:
            break
    accepted.sort(key=lambda item: (item["target_id"], item["relation"], item["evidence_line"]))
    return accepted, rejected


def _to_edge_rows(
    repo_id: str,
    path: str,
    edges: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    model: str,
    file_hash: str,
) -> list[dict[str, Any]]:
    by_id = {str(item["id"]): item for item in candidates}
    source_key = symbol_key(path, MODULE_QUALNAME)
    rows: list[dict[str, Any]] = []
    for edge in edges:
        target_id = str(edge.get("target_id") or "")
        candidate = by_id.get(target_id)
        if candidate is None:
            # 缓存命中的旧 target 可能已不在当前候选集（导入变化），跳过。
            continue
        relation = str(edge.get("relation") or "").upper()
        if relation not in ALLOWED_RELATIONS:
            continue
        target_key = symbol_key(str(candidate["path"]), str(candidate["qualname"]))
        line = int(edge.get("evidence_line") or 0)
        rows.append(
            {
                "type": relation,
                "repo_id": repo_id,
                "from_label": "Symbol",
                "from_kind": "module",
                "from_qualname": source_key,
                "to_label": "Symbol",
                "to_kind": str(candidate["kind"]),
                "to_qualname": target_key,
                "edge_key": f"{relation}:{source_key}->{target_key}@{line}",
                "props": {
                    "relation": relation,
                    "evidence_line": line,
                    "lineno": line,
                    "confidence": float(edge.get("confidence") or 0.0),
                    "source": "llm",
                    "model": model,
                    "file_hash": file_hash,
                    "target": target_id,
                },
            }
        )
    return rows


def extract_llm_edges(
    store: Any,
    repo_id: str,
    llm: Any,
    *,
    config: LlmEdgeConfig | None = None,
    cache: LlmEdgeCache | None = None,
    data: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    repo: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """对 repo 内文件抽取 LLM 候选边；返回 edges/stats/skipped/cache。"""
    started = perf_counter()
    resolved_config = config or LlmEdgeConfig()
    repo_row = dict(repo or store.get_repo(repo_id) or {})
    payload = dict(data or store.export_repo(repo_id))
    cache_store = cache or LlmEdgeCache(store.db_path)
    model = model_label(llm)
    module_index = build_module_index(payload.get("files") or [])
    files = sorted(
        (item for item in payload.get("files") or [] if item.get("path")),
        key=lambda item: normalize_path(str(item.get("path") or "")),
    )
    wanted = set(resolved_config.only_paths or ())
    if wanted:
        files = [item for item in files if normalize_path(str(item.get("path") or "")) in wanted]
    if resolved_config.max_files is not None:
        files = files[: max(0, int(resolved_config.max_files))]

    stats: dict[str, Any] = {
        "files": len(files),
        "candidates": 0,
        "cache_hits": 0,
        "llm_calls": 0,
        "edges": 0,
        "rejected": 0,
        "skipped": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    skipped: list[dict[str, str]] = []
    edge_rows: list[dict[str, Any]] = []
    rejects_total = {"not_candidate": 0, "bad_relation": 0, "low_confidence": 0, "bad_line": 0, "duplicate": 0}

    for item in files:
        path = normalize_path(str(item.get("path") or ""))
        file_hash = str(item.get("file_hash") or "")
        if not file_hash:
            skipped.append({"path": path, "reason": "missing_file_hash"})
            stats["skipped"] += 1
            continue
        candidates = build_candidates(path, payload, module_index, resolved_config)
        stats["candidates"] += len(candidates)
        if not candidates:
            skipped.append({"path": path, "reason": "no_candidates"})
            stats["skipped"] += 1
            continue
        cached = cache_store.get(repo_id, file_hash, model)
        if cached is not None:
            stats["cache_hits"] += 1
            normalized = cached
        else:
            code = _file_code(path, payload)
            path_chunks = [
                chunk
                for chunk in payload.get("chunks") or []
                if normalize_path(str(chunk.get("path") or "")) == path
            ]
            loc = max([int(chunk.get("end_line") or 0) for chunk in path_chunks] or [0])
            messages = build_messages(path, code, candidates, resolved_config)
            if messages is None:
                skipped.append({"path": path, "reason": "token_budget"})
                stats["skipped"] += 1
                continue
            try:
                stats["llm_calls"] += 1
                response = llm.chat(
                    messages,
                    max_tokens=resolved_config.max_output_tokens,
                    json_mode=True,
                )
                stats["tokens_in"] += int(response.get("tokens_in") or 0)
                stats["tokens_out"] += int(response.get("tokens_out") or 0)
                normalized, rejects = parse_llm_edges(
                    str(response.get("text") or ""), candidates, resolved_config, loc=loc
                )
            except Exception as exc:  # noqa: BLE001 - 单文件失败跳过并记录
                skipped.append({"path": path, "reason": f"llm_error:{type(exc).__name__}: {str(exc)[:160]}"})
                stats["skipped"] += 1
                continue
            for key, value in rejects.items():
                rejects_total[key] += value
            cache_store.put(repo_id, file_hash, model, path, normalized)
        edge_rows.extend(
            _to_edge_rows(repo_id, path, normalized, candidates, model=model, file_hash=file_hash)
        )

    stats["edges"] = len(edge_rows)
    stats["rejected"] = sum(rejects_total.values())
    stats["reject_reasons"] = rejects_total
    stats["cache"] = cache_store.stats(repo_id)
    stats["ms"] = round((perf_counter() - started) * 1000, 1)
    stats["model"] = model
    stats["repo"] = str(repo_row.get("repo_id") or repo_id)
    return {
        "edges": edge_rows,
        "stats": stats,
        "skipped": skipped,
        "cache": stats["cache"],
    }


__all__ = [
    "ALLOWED_RELATIONS",
    "BUILTIN_NAMES",
    "LlmEdgeCache",
    "LlmEdgeConfig",
    "build_candidates",
    "build_messages",
    "eligible_name",
    "estimate_tokens",
    "extract_llm_edges",
    "model_label",
    "parse_llm_edges",
]
