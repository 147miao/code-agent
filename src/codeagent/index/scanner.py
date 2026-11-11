"""Python 源码仓库扫描。

职责：
- 递归遍历 root 下的 `.py` 文件；
- 排除 `.git`、`.venv*`、`__pycache__`、`site-packages`、`data`、`.volumes`、
  `node_modules`、`dist`、`build`；
- 校验 root.resolve() 位于某个 allowed_root 下，不跟随目录符号链接、不索引文件符号链接，
  防止符号链接逃逸；
- 计算 sha1 file_hash；单文件 >1MB 或 >5000 行时跳过并记录（不读取入库）。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "site-packages",
        "data",
        ".volumes",
        "node_modules",
        "dist",
        "build",
    }
)
EXCLUDED_DIR_PREFIXES = (".venv",)
MAX_FILE_BYTES = 1_000_000
MAX_FILE_LINES = 5000
HASH_CHUNK_SIZE = 1024 * 1024

REASON_SYMLINK = "symlink"
REASON_TOO_LARGE = "file_too_large"
REASON_TOO_MANY_LINES = "too_many_lines"
REASON_UNREADABLE = "unreadable"


class RootNotAllowedError(PermissionError):
    """root 解析后不在 allowed_roots 范围内（或符号链接逃逸）。"""


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """一个待索引文件；path 为相对 root 的 POSIX 路径。"""

    path: str
    abs_path: Path
    file_hash: str
    size: int
    mtime: float
    line_count: int


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """被扫描器跳过的文件及原因。"""

    path: str
    reason: str


def file_hash(path: str | Path) -> str:
    """流式计算文件 sha1（与 scanner 入库值一致）。"""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(HASH_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_lines(data: bytes) -> int:
    """按字节统计行数：空文件 0 行，末行无换行也算一行。"""
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def is_excluded_dir(name: str) -> bool:
    """目录名是否命中排除规则（.venv 支持任意后缀）。"""
    return name in EXCLUDED_DIR_NAMES or name.startswith(EXCLUDED_DIR_PREFIXES)


def resolve_root(root: str | Path, allowed_roots: list[str | Path] | None = None) -> Path:
    """解析并校验 root；越界抛 RootNotAllowedError。

    root 必须真实存在且为目录；resolve() 会展开符号链接，因此指向 allowed_root 之外的
    符号链接 root 会被拒绝。allowed_roots=None 时使用当前用户 home（与计划默认一致）。
    """
    candidate = Path(root).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"root 不存在: {candidate}") from None
    if not resolved.is_dir():
        raise NotADirectoryError(f"root 不是目录: {resolved}")
    roots = [Path.home()] if allowed_roots is None else [Path(item).expanduser() for item in allowed_roots]
    for allowed in roots:
        allowed_resolved = allowed.resolve()
        if resolved == allowed_resolved or resolved.is_relative_to(allowed_resolved):
            return resolved
    allowed_text = ", ".join(str(Path(item).expanduser().resolve()) for item in roots) or "<empty>"
    raise RootNotAllowedError(f"root 不在 allowed_roots 内: {resolved}（allowed: {allowed_text}）")


def scan_repo(
    root: str | Path,
    allowed_roots: list[str | Path] | None = None,
    *,
    max_file_bytes: int | None = None,
    max_file_lines: int | None = None,
) -> tuple[list[ScannedFile], list[SkippedFile]]:
    """扫描 root，返回 (待索引文件, 跳过文件) 两个有序列表。

    max_file_bytes / max_file_lines 默认使用模块常量，测试可传小值。
    """
    resolved_root = resolve_root(root, allowed_roots)
    byte_limit = MAX_FILE_BYTES if max_file_bytes is None else int(max_file_bytes)
    line_limit = MAX_FILE_LINES if max_file_lines is None else int(max_file_lines)
    files: list[ScannedFile] = []
    skipped: list[SkippedFile] = []

    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not is_excluded_dir(name))
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            abs_path = Path(dirpath) / filename
            rel_path = abs_path.relative_to(resolved_root).as_posix()
            if abs_path.is_symlink():
                skipped.append(SkippedFile(rel_path, REASON_SYMLINK))
                continue
            try:
                stat = abs_path.stat()
            except OSError:
                skipped.append(SkippedFile(rel_path, REASON_UNREADABLE))
                continue
            if stat.st_size > byte_limit:
                skipped.append(SkippedFile(rel_path, REASON_TOO_LARGE))
                continue
            try:
                data = abs_path.read_bytes()
            except OSError:
                skipped.append(SkippedFile(rel_path, REASON_UNREADABLE))
                continue
            line_count = count_lines(data)
            if line_count > line_limit:
                skipped.append(SkippedFile(rel_path, REASON_TOO_MANY_LINES))
                continue
            files.append(
                ScannedFile(
                    path=rel_path,
                    abs_path=abs_path,
                    file_hash=hashlib.sha1(data).hexdigest(),
                    size=len(data),
                    mtime=float(stat.st_mtime),
                    line_count=line_count,
                )
            )

    files.sort(key=lambda item: item.path)
    skipped.sort(key=lambda item: (item.path, item.reason))
    return files, skipped
