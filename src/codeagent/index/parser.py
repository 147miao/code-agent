"""stdlib ast 解析器：模块、import、名称引用与符号树。

约定：
- 每个模块产生一个 kind="module"、qualname="<module>" 的符号，覆盖文件首行到末行；
- 顶层类/函数 qualname 就是名字；嵌套用点号：`Class.method`、`outer.inner`；
- 类体内直接定义的函数 kind="method"；函数内再嵌套的函数 kind="function"；
- start_line 取 decorator 与 def/class 的最小行号（因此 code 含 decorator），end_line 取
  ast 的 end_lineno；parent 为外层符号 qualname，顶层为 "<module>"；
- 语法/解码错误写入 ModuleInfo.error，调用方跳过该文件。
"""
from __future__ import annotations

import ast
import os
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODULE_QUALNAME = "<module>"
SYMBOL_KINDS = frozenset({"module", "class", "function", "method"})


@dataclass(slots=True)
class ImportRef:
    module: str
    level: int
    lineno: int

    def as_dict(self) -> dict[str, Any]:
        return {"module": self.module, "level": self.level, "lineno": self.lineno}


@dataclass(slots=True)
class Reference:
    name: str
    lineno: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lineno": self.lineno}


@dataclass(slots=True)
class Symbol:
    qualname: str
    kind: str
    start_line: int
    end_line: int
    docstring: str | None
    code: str
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    parent: str | None = None


@dataclass(slots=True)
class ModuleInfo:
    path: str
    imports: list[ImportRef] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    error: str | None = None


def parse_file(path: str | Path, *, path_label: str | None = None) -> ModuleInfo:
    """读取并解析一个 .py 文件；path_label 用于把绝对路径映射为仓库相对路径。"""
    label = os.fspath(path) if path_label is None else path_label
    try:
        with tokenize.open(path) as handle:
            source = handle.read()
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        return ModuleInfo(path=label, error=f"{type(exc).__name__}: {exc}")
    return parse_source(label, source)


def parse_source(path: str | Path, source: str) -> ModuleInfo:
    """解析源码字符串；语法错误不抛异常，返回 error 的 ModuleInfo。"""
    label = os.fspath(path)
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as exc:
        detail = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return ModuleInfo(path=label, error=detail)
    except ValueError as exc:
        return ModuleInfo(path=label, error=f"ValueError: {exc}")

    lines = source.splitlines()
    symbols = _collect_symbols(tree, lines)
    return ModuleInfo(
        path=label,
        imports=_collect_imports(tree),
        references=_collect_references(tree),
        symbols=symbols,
    )


def _node_start_line(node: ast.AST) -> int:
    """def/class 起始行；有 decorator 时取最靠上的 decorator 行。"""
    start = int(getattr(node, "lineno", 1) or 1)
    for decorator in getattr(node, "decorator_list", []) or []:
        start = min(start, int(getattr(decorator, "lineno", start) or start))
    return start


def _node_end_line(node: ast.AST, fallback: int) -> int:
    return int(getattr(node, "end_lineno", None) or fallback)


def _code_slice(lines: list[str], start_line: int, end_line: int) -> str:
    """1-based 闭区间源码切片；无内容返回空串。"""
    if not lines or end_line < start_line:
        return ""
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)
    return "\n".join(lines[start:end])


def _decorator_text(node: ast.AST) -> str:
    return ast.unparse(node)


def _collect_symbols(tree: ast.Module, lines: list[str]) -> list[Symbol]:
    module_end = max(1, len(lines))
    symbols: list[Symbol] = [
        Symbol(
            qualname=MODULE_QUALNAME,
            kind="module",
            start_line=1,
            end_line=module_end,
            docstring=ast.get_docstring(tree, clean=True),
            code=_code_slice(lines, 1, module_end),
            parent=None,
        )
    ]
    _walk_scope(tree, MODULE_QUALNAME, "module", lines, symbols)
    return symbols


def _walk_scope(
    container: ast.AST,
    scope_qualname: str,
    scope_kind: str,
    lines: list[str],
    symbols: list[Symbol],
) -> None:
    """递归收集符号；if/try/with 等语句不产生 qualname 段。"""
    for child in ast.iter_child_nodes(container):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = child.name if scope_qualname == MODULE_QUALNAME else f"{scope_qualname}.{child.name}"
            kind = "method" if scope_kind == "class" else "function"
            start_line = _node_start_line(child)
            end_line = _node_end_line(child, start_line)
            symbols.append(
                Symbol(
                    qualname=qualname,
                    kind=kind,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=ast.get_docstring(child, clean=True),
                    code=_code_slice(lines, start_line, end_line),
                    decorators=[_decorator_text(item) for item in child.decorator_list],
                    is_async=isinstance(child, ast.AsyncFunctionDef),
                    parent=scope_qualname,
                )
            )
            _walk_scope(child, qualname, kind, lines, symbols)
        elif isinstance(child, ast.ClassDef):
            qualname = child.name if scope_qualname == MODULE_QUALNAME else f"{scope_qualname}.{child.name}"
            start_line = _node_start_line(child)
            end_line = _node_end_line(child, start_line)
            symbols.append(
                Symbol(
                    qualname=qualname,
                    kind="class",
                    start_line=start_line,
                    end_line=end_line,
                    docstring=ast.get_docstring(child, clean=True),
                    code=_code_slice(lines, start_line, end_line),
                    decorators=[_decorator_text(item) for item in child.decorator_list],
                    is_async=False,
                    parent=scope_qualname,
                )
            )
            _walk_scope(child, qualname, "class", lines, symbols)
        else:
            _walk_scope(child, scope_qualname, scope_kind, lines, symbols)


def _collect_imports(tree: ast.Module) -> list[ImportRef]:
    imports: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportRef(module=alias.name, level=0, lineno=int(node.lineno)))
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                ImportRef(module=node.module or "", level=int(node.level or 0), lineno=int(node.lineno))
            )
    imports.sort(key=lambda item: (item.lineno, item.module, item.level))
    return imports


def _collect_references(tree: ast.Module) -> list[Reference]:
    """收集所有 Load 上下文的 ast.Name（去重后按行号、名字排序）。"""
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            seen.add((int(node.lineno), node.id))
    return [Reference(name=name, lineno=lineno) for lineno, name in sorted(seen)]
