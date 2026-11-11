"""解析器测试：真实新包结构、嵌套/async/decorator、import/reference、语法错误。"""
from __future__ import annotations

from pathlib import Path

from codeagent.index.parser import MODULE_QUALNAME, ModuleInfo, parse_file, parse_source

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "src" / "codeagent" / "index" / "store.py"

NESTED_SOURCE = '''"""module doc."""

import os
from . import sibling
from ..pkg import mod as m


def outer(x):
    """outer doc."""
    @decorator
    async def inner():
        class Local:
            def method(self):
                return os.path

        return Local

    return inner


@staticmethod
class Meta:
    @classmethod
    async def make(cls):
        return m
'''


def _by_qualname(info: ModuleInfo) -> dict:
    return {symbol.qualname: symbol for symbol in info.symbols}


def _line_of(source_lines: list[str], needle: str, *, start: int = 0) -> int:
    for index in range(start, len(source_lines)):
        if needle in source_lines[index]:
            return index + 1
    raise AssertionError(f"源码中找不到: {needle}")


def test_parse_real_store_structure_and_line_numbers() -> None:
    info = parse_file(STORE_PATH, path_label="src/codeagent/index/store.py")
    assert info.error is None
    assert info.path == "src/codeagent/index/store.py"
    source_lines = STORE_PATH.read_text(encoding="utf-8").splitlines()
    symbols = _by_qualname(info)

    module = symbols[MODULE_QUALNAME]
    assert module.kind == "module"
    assert module.parent is None
    assert (module.start_line, module.end_line) == (1, len(source_lines))
    assert module.docstring is not None and module.docstring.startswith("SQLite + FTS5")

    store_class = symbols["CodeStore"]
    class_line = _line_of(source_lines, "class CodeStore:")
    assert store_class.kind == "class"
    assert store_class.parent == MODULE_QUALNAME
    assert store_class.start_line == class_line
    assert store_class.end_line >= store_class.start_line
    assert source_lines[store_class.start_line - 1].startswith("class CodeStore:")

    search = symbols["CodeStore.search"]
    search_line = _line_of(source_lines, "    def search(", start=class_line - 1)
    assert search.kind == "method"
    assert search.parent == "CodeStore"
    assert search.start_line == search_line
    assert search.end_line >= search.start_line
    assert source_lines[search.start_line - 1].strip().startswith("def search(")

    fusion = symbols["reciprocal_rank_fusion"]
    assert fusion.kind == "function"
    assert fusion.start_line == _line_of(source_lines, "def reciprocal_rank_fusion(")

    imports = {(item.module, item.level, item.lineno) for item in info.imports}
    assert ("sqlite3", 0, _line_of(source_lines, "import sqlite3")) in imports
    assert ("pathlib", 0, _line_of(source_lines, "from pathlib import Path")) in imports
    assert ("collections.abc", 0, _line_of(source_lines, "from collections.abc import")) in imports

    references = {(item.name, item.lineno) for item in info.references}
    assert ("Path", _line_of(source_lines, "self.db_path = Path(db_path)")) in references


def test_parse_nested_async_decorated_symbols() -> None:
    info = parse_source("pkg/nested.py", NESTED_SOURCE)
    assert info.error is None
    symbols = _by_qualname(info)

    outer = symbols["outer"]
    assert outer.kind == "function" and outer.parent == MODULE_QUALNAME
    assert outer.docstring == "outer doc."
    assert (outer.start_line, outer.end_line) == (8, 18)

    inner = symbols["outer.inner"]
    assert inner.kind == "function" and inner.is_async is True
    assert inner.parent == "outer"
    assert inner.decorators == ["decorator"]
    assert inner.start_line == 10  # decorator 行
    assert inner.code.startswith("    @decorator")

    local = symbols["outer.inner.Local"]
    assert local.kind == "class" and local.parent == "outer.inner"
    method = symbols["outer.inner.Local.method"]
    assert method.kind == "method" and method.parent == "outer.inner.Local"

    meta = symbols["Meta"]
    assert meta.kind == "class" and meta.decorators == ["staticmethod"]
    assert meta.start_line == 21
    make = symbols["Meta.make"]
    assert make.kind == "method" and make.is_async is True
    assert make.decorators == ["classmethod"]
    assert make.parent == "Meta"

    imports = {(item.module, item.level, item.lineno) for item in info.imports}
    assert ("os", 0, 3) in imports
    assert ("", 1, 4) in imports
    assert ("pkg", 2, 5) in imports

    references = {(item.name, item.lineno) for item in info.references}
    assert ("os", 14) in references
    assert ("m", 25) in references
    assert ("decorator", 10) in references


def test_syntax_error_returns_error_and_skips_symbols(tmp_path: Path) -> None:
    info = parse_source("bad.py", "def broken(:\n")
    assert info.error is not None and "SyntaxError" in info.error
    assert info.symbols == [] and info.imports == []

    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class (:\n", encoding="utf-8")
    from_file = parse_file(bad_file, path_label="bad.py")
    assert from_file.error is not None and "SyntaxError" in from_file.error
    assert from_file.symbols == []


def test_empty_module_still_has_module_symbol() -> None:
    info = parse_source("empty.py", "")
    assert info.error is None
    assert len(info.symbols) == 1
    module = info.symbols[0]
    assert (module.qualname, module.kind) == (MODULE_QUALNAME, "module")
    assert (module.start_line, module.end_line) == (1, 1)
    assert module.code == ""
    assert module.docstring is None
