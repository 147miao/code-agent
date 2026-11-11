"""代码库索引基础组件（Python AST + SQLite FTS5）。

模块：
- scanner：目录扫描、排除规则、allowed_roots 越界保护、文件指纹；
- parser：stdlib ast 解析模块/类/函数/方法/嵌套符号与 import/reference；
- chunker：符号到检索块（超长按行切分，确定性 chunk_id）；
- schema：SQLite 表 / 双 FTS5 / 同步触发器 DDL；
- store：SQLite（WAL + unicode61/trigram 双 FTS5 + RRF）持久化与检索；
- indexer：扫描 → 解析 → 切块 → 入库的增量索引器。

仅支持 Python；反向 import 为文件级启发式，不承诺精确调用链。
"""
from __future__ import annotations

from .chunker import MAX_CHUNK_CHARS, MAX_CHUNK_LINES, build_chunks, make_chunk_id
from .indexer import CodeIndexer
from .parser import MODULE_QUALNAME, ImportRef, ModuleInfo, Reference, Symbol, parse_file, parse_source
from .scanner import (
    MAX_FILE_BYTES,
    MAX_FILE_LINES,
    REASON_SYMLINK,
    REASON_TOO_LARGE,
    REASON_TOO_MANY_LINES,
    REASON_UNREADABLE,
    RootNotAllowedError,
    ScannedFile,
    SkippedFile,
    count_lines,
    file_hash,
    resolve_root,
    scan_repo,
)
from .store import CodeStore, reciprocal_rank_fusion

__all__ = [
    "MAX_CHUNK_CHARS",
    "MAX_CHUNK_LINES",
    "MAX_FILE_BYTES",
    "MAX_FILE_LINES",
    "MODULE_QUALNAME",
    "REASON_SYMLINK",
    "REASON_TOO_LARGE",
    "REASON_TOO_MANY_LINES",
    "REASON_UNREADABLE",
    "CodeIndexer",
    "CodeStore",
    "ImportRef",
    "ModuleInfo",
    "Reference",
    "RootNotAllowedError",
    "ScannedFile",
    "SkippedFile",
    "Symbol",
    "build_chunks",
    "count_lines",
    "file_hash",
    "make_chunk_id",
    "parse_file",
    "parse_source",
    "reciprocal_rank_fusion",
    "resolve_root",
    "scan_repo",
]
