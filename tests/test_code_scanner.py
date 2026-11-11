"""扫描器测试：排除规则、allowed_roots、符号链接逃逸、大小/行数限制与 hash。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from codeagent.index.scanner import (
    REASON_SYMLINK,
    REASON_TOO_LARGE,
    REASON_TOO_MANY_LINES,
    RootNotAllowedError,
    count_lines,
    file_hash,
    resolve_root,
    scan_repo,
)

EXCLUDED_DIRS = (
    ".git",
    ".venv",
    ".venv-test",
    "__pycache__",
    "site-packages",
    "data",
    ".volumes",
    "node_modules",
    "dist",
    "build",
)


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_scan_collects_python_and_excludes_directories(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "pkg" / "a.py", "def a():\n    return 1\n")
    _write(root / "b.py", "B = 2\n")
    _write(root / "notes.txt", "not python\n")
    for name in EXCLUDED_DIRS:
        _write(root / name / "hidden.py", "HIDDEN = 1\n")

    files, skipped = scan_repo(root, [tmp_path])

    assert [item.path for item in files] == ["b.py", "pkg/a.py"]
    assert skipped == []
    first = files[1]
    assert first.abs_path == root / "pkg" / "a.py"
    assert first.file_hash == hashlib.sha1(first.abs_path.read_bytes()).hexdigest()
    assert first.size == first.abs_path.stat().st_size
    assert first.line_count == 2
    assert file_hash(first.abs_path) == first.file_hash


def test_scan_skips_large_and_long_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "ok.py", "x = 1\n")
    _write(root / "big.py", "y = 2\n" * 20)
    _write(root / "long.py", "".join(f"v{i} = {i}\n" for i in range(5)))

    files, skipped = scan_repo(root, [tmp_path], max_file_bytes=50, max_file_lines=3)

    assert [item.path for item in files] == ["ok.py"]
    reasons = {item.path: item.reason for item in skipped}
    assert reasons["big.py"] == REASON_TOO_LARGE
    assert reasons["long.py"] == REASON_TOO_MANY_LINES


def test_root_outside_allowed_roots_is_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    _write(outside / "a.py")

    with pytest.raises(RootNotAllowedError):
        resolve_root(outside, [allowed])
    with pytest.raises(PermissionError):
        scan_repo(outside, [allowed])


def test_symlink_root_escape_is_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    _write(outside / "secret.py")
    link = allowed / "linked"
    os.symlink(outside, link)

    with pytest.raises(RootNotAllowedError):
        resolve_root(link, [allowed])


def test_symlink_file_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    _write(outside)
    os.symlink(outside, root / "escape.py")
    _write(root / "real.py")

    files, skipped = scan_repo(root, [tmp_path])

    assert [item.path for item in files] == ["real.py"]
    assert [(item.path, item.reason) for item in skipped] == [("escape.py", REASON_SYMLINK)]


def test_missing_root_and_count_lines(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_root(tmp_path / "missing", [tmp_path])
    assert count_lines(b"") == 0
    assert count_lines(b"a") == 1
    assert count_lines(b"a\n") == 1
    assert count_lines(b"a\nb") == 2
