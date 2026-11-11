"""代码库分析 E2E（hermetic）：自举索引 codeagent，验证检索命中与符号行号。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagent.index.indexer import CodeIndexer
from codeagent.index.store import CodeStore

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((Path(__file__).parent / "code_e2e_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    db = tmp_path_factory.mktemp("code") / "index.sqlite"
    report = CodeIndexer(db, allowed_roots=[str(ROOT.parent)]).index(str(ROOT), "codeagent")
    store = CodeStore(db)
    return store, report["repo_id"], report


def test_self_index_has_symbols(indexed):
    store, repo_id, report = indexed
    assert report["files"] >= 12
    assert report["symbols"] >= 100
    assert report["chunks"] >= 100
    assert store.get_repo(repo_id) is not None


def test_overview_has_kinds_and_top_paths(indexed):
    store, repo_id, _ = indexed
    overview = store.overview(repo_id)
    assert {"class", "function", "method"} <= set(overview["kinds"]), overview["kinds"]
    top_paths = overview["top_paths"]
    assert top_paths, "top_paths must not be empty"
    assert {"path", "symbols"} <= set(top_paths[0])
    assert top_paths[0]["path"].endswith(".py")
    assert top_paths[0]["symbols"] >= top_paths[-1]["symbols"]


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:18] for c in CASES])
def test_search_hits_expected_path(indexed, case):
    store, repo_id, _ = indexed
    hits = store.search(case["question"], k=5, repo_id=repo_id)
    paths = [hit["path"] for hit in hits]
    assert any(expected in path for path in paths for expected in case["expect_paths"]), paths
    quals = [hit["qualname"] for hit in hits]
    assert any(
        qualname in qual or qual.startswith(qualname)
        for qualname in case.get("expect_qualnames", [])
        for qual in quals
    ), quals


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:18] for c in CASES])
def test_symbol_lookup_has_lines(indexed, case):
    store, repo_id, _ = indexed
    for qualname in case["expect_qualnames"]:
        row = store.get_symbol(repo_id, qualname)
        assert row is not None, qualname
        assert row["start_line"] >= 1 and row["end_line"] >= row["start_line"]
        assert row["path"].endswith(".py")
