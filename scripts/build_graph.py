"""索引本地 Python 仓库并构建 Neo4j 代码图。

用法：
    python scripts/build_graph.py --root . --name codeagent --reindex
    python scripts/build_graph.py --root . --reindex --with-llm --llm-path src/codeagent/api/routes.py

说明：
- 先做增量 SQLite 索引（可选 --reindex），再从 CodeStore 导出 AST 数据写 Neo4j；
- Neo4j 连接读取 NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE；
- --with-llm 用仓库 .env 里的 OPENCODE_GO_API_KEY 调用 FallbackLLM 生成 CALLS/TESTS；
- --llm-path 可重复，只对这些文件调用 LLM（AST 边仍全量重建）；
- 退出码：0 成功，2 参数/配置/连接错误。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from codeagent.codegraph import GraphService
from codeagent.config import Settings
from codeagent.index.indexer import CodeIndexer
from codeagent.index.store import CodeStore
from codeagent.llm import build_llm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_NAME = "code_index.sqlite"


def default_db_path() -> Path:
    configured = os.environ.get("CODEAGENT_DATA_ROOT", "").strip()
    base = Path(configured).expanduser() if configured else REPO_ROOT / "data"
    return base / DEFAULT_DB_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="索引仓库并构建 Neo4j 代码图")
    parser.add_argument("--root", required=True, help="待索引仓库根目录")
    parser.add_argument("--name", default="", help="仓库名（默认取目录名）")
    parser.add_argument("--db", default=str(default_db_path()), help="SQLite 索引路径")
    parser.add_argument("--reindex", action="store_true", help="先全量重建 SQLite 索引")
    parser.add_argument("--with-llm", action="store_true", help="追加 LLM CALLS/TESTS 候选边")
    parser.add_argument("--llm-path", action="append", default=None, help="只对这些文件调用 LLM（可重复）")
    parser.add_argument("--allow-root", action="append", default=None, help="允许索引的根目录（默认 home）")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印完整构建报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser()
    allowed = [Path(item).expanduser() for item in args.allow_root] if args.allow_root else None
    settings = Settings()
    if not settings.graph_configured:
        print("error: NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD 未配置", file=sys.stderr)
        return 2

    indexer = CodeIndexer(Path(args.db).expanduser(), allowed)
    try:
        index_report = indexer.index(str(root), args.name or root.name, reindex=bool(args.reindex))
    except (PermissionError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    store = CodeStore(Path(args.db).expanduser())
    llm = build_llm(settings) if args.with_llm else None
    if llm is not None and not getattr(llm, "providers", None):
        print("error: 没有可用的 LLM provider（检查 OPENCODE_GO_API_KEY）", file=sys.stderr)
        return 2
    service = GraphService(settings, store, llm=llm)
    try:
        report = service.build(
            str(index_report["repo_id"]),
            bool(args.with_llm),
            llm_paths=args.llm_path,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 统一错误出口
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    result = {"index": index_report, "graph": report}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"repo: {index_report['repo_id']}")
        print(f"root: {root}")
        print(f"files: {index_report['files']} symbols: {index_report['symbols']} chunks: {index_report['chunks']}")
        print(f"nodes: {report['total_nodes']} edges: {report['total_edges']} ms: {report['ms']}")
        print(f"  nodes: {report['nodes']}")
        print(f"  edges: {report['edges']}")
        if report.get("llm"):
            print(f"  llm: {json.dumps(report['llm'], ensure_ascii=False)}")
        if report.get("llm_skipped"):
            print(f"  llm_skipped: {len(report['llm_skipped'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
