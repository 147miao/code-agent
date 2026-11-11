"""增量代码索引器：扫描 → 解析 → 切块 → SQLite。

增量规则：
- 文件 hash 未变：跳过解析；
- hash 变化/新文件：先删除该文件全部旧行，再解析并重新插入；
- 已删除文件（含被大小/行数限制跳过）：清理旧行；
- 语法错误文件不写 files 行，下次索引会重试并在 report.errors 中再次报告；
- reindex=True 先清空该仓库全部派生行，再全量重建。
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .chunker import build_chunks
from .parser import ModuleInfo, parse_file
from .scanner import ScannedFile, SkippedFile, resolve_root, scan_repo
from .store import CodeStore

logger = logging.getLogger(__name__)


class CodeIndexer:
    """面向单个 root 的代码索引器；allowed_roots 用于阻止任意目录读取。

    allowed_roots=None（省略）使用 [home]；显式 [] 表示全部拒绝，与 API 路由对
    CODE_ALLOWED_ROOTS=[] 的语义一致。
    """

    def __init__(self, db_path: str | Path, allowed_roots: list[str | Path] | None = None) -> None:
        self.db_path = Path(db_path)
        if allowed_roots is None:
            self.allowed_roots = [Path.home()]
        else:
            self.allowed_roots = [Path(item) for item in allowed_roots]
        self.store = CodeStore(self.db_path)

    def index(self, root: str, name: str, reindex: bool = False) -> dict[str, Any]:
        """索引 root 并返回 report；root 越界抛 RootNotAllowedError（PermissionError）。"""
        started = perf_counter()
        root_path = resolve_root(root, self.allowed_roots)
        repo = self.store.add_repo(str(root_path), name or root_path.name)
        repo_id = str(repo["repo_id"])
        if reindex:
            self.store.reindex_repo(repo_id)

        existing = {str(item["path"]): str(item["file_hash"]) for item in self.store.list_files(repo_id)}
        scanned, skipped = scan_repo(root_path, self.allowed_roots)
        errors: list[dict[str, str]] = []
        seen: set[str] = set()
        try:
            for item in scanned:
                seen.add(item.path)
                if not reindex and existing.get(item.path) == item.file_hash:
                    continue
                self.store.delete_file_rows(repo_id, item.path)
                info = parse_file(item.abs_path, path_label=item.path)
                if info.error:
                    errors.append({"path": item.path, "error": info.error})
                    continue
                self._write_file(repo_id, item, info)
            for deleted_path in sorted(set(existing) - seen):
                self.store.delete_file_rows(repo_id, deleted_path)
            overview = self.store.overview(repo_id)
            self.store.update_repo_stats(
                repo_id,
                file_count=int(overview.get("file_count", 0)),
                symbol_count=int(overview.get("symbol_count", 0)),
                status="ready",
            )
        except Exception:
            self._mark_error(repo_id)
            raise

        elapsed_ms = round((perf_counter() - started) * 1000, 1)
        return {
            "repo_id": repo_id,
            "files": int(overview.get("file_count", 0)),
            "symbols": int(overview.get("symbol_count", 0)),
            "imports": int(overview.get("import_count", 0)),
            "references": int(overview.get("reference_count", 0)),
            "chunks": int(overview.get("chunk_count", 0)),
            "skipped": [_skipped_dict(item) for item in skipped],
            "errors": errors,
            "ms": elapsed_ms,
        }

    def _write_file(self, repo_id: str, item: ScannedFile, info: ModuleInfo) -> None:
        self.store.upsert_file(repo_id, item.path, item.file_hash, item.size, item.mtime)
        self.store.insert_symbols(repo_id, item.path, [asdict(symbol) for symbol in info.symbols])
        self.store.insert_imports(repo_id, item.path, [asdict(import_ref) for import_ref in info.imports])
        self.store.insert_references(repo_id, item.path, [asdict(ref) for ref in info.references])
        chunks = build_chunks(repo_id, item.path, info.symbols)
        self.store.insert_chunks(repo_id, chunks)

    def _mark_error(self, repo_id: str) -> None:
        try:
            overview = self.store.overview(repo_id)
            self.store.update_repo_stats(
                repo_id,
                file_count=int(overview.get("file_count", 0)),
                symbol_count=int(overview.get("symbol_count", 0)),
                status="error",
            )
        except sqlite3.Error:  # pragma: no cover - 标记失败不应覆盖原始异常
            logger.warning("failed to mark repo %s as error", repo_id, exc_info=True)


def _skipped_dict(item: SkippedFile) -> dict[str, str]:
    return {"path": item.path, "reason": item.reason}
