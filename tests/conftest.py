"""共享测试夹具：临时仓库、索引服务、会话库、配置。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from codeagent.config import Settings
from codeagent.index.indexer import CodeIndexer
from codeagent.index.store import CodeStore
from codeagent.retrieval.service import CodeService
from codeagent.sessions.repository import SQLiteSessionRepository


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_repo(root: Path) -> Path:
    write_text(root / "pkg" / "__init__.py", "")
    write_text(
        root / "pkg" / "b.py",
        '"""Demo module."""\n\n\n'
        "class ToolExecutor:\n"
        '    """Run tools."""\n\n'
        "    def run(self, name):\n"
        "        return name\n",
    )
    write_text(
        root / "pkg" / "a.py",
        "from .b import ToolExecutor\n\n\ndef make():\n    return ToolExecutor()\n",
    )
    return root


def build_index(root: Path, tmp_path: Path) -> SimpleNamespace:
    db = tmp_path / "code.sqlite"
    indexer = CodeIndexer(db, [str(tmp_path)])
    report = indexer.index(str(root), "demo")
    store = CodeStore(db)
    return SimpleNamespace(
        root=root,
        db=db,
        store=store,
        service=CodeService(store),
        repo_id=str(report["repo_id"]),
        report=report,
        indexer=indexer,
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def indexed(tmp_path: Path, sample_repo: Path) -> SimpleNamespace:
    return build_index(sample_repo, tmp_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path / "data",
        db_path="",
        session_db_path="",
        database_url="",
        code_allowed_roots=[str(tmp_path)],
    )


@pytest.fixture
def sessions(tmp_path: Path) -> SQLiteSessionRepository:
    return SQLiteSessionRepository(tmp_path / "sessions.sqlite")
