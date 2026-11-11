"""索引本地 Python 代码库到 SQLite FTS5。

用法：
    python scripts/index_code.py --root /path/to/repo --name myrepo
    python scripts/index_code.py --root . --reindex --json

说明：
- 只支持 Python；检索为 unicode61 + trigram 双 FTS5（RRF 合并）；
- --db 默认是 $CODEAGENT_DATA_ROOT/code_index.sqlite（未设置时用仓库 data/code_index.sqlite）；
- --root 必须位于 --allow-root 指定的目录内（默认当前用户 home；可重复传 --allow-root）；
- 退出码：0 成功，2 参数/权限/路径错误。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from codeagent.index.indexer import CodeIndexer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_NAME = "code_index.sqlite"


def default_data_root() -> Path:
    """数据根目录：CODEAGENT_DATA_ROOT 优先，否则仓库内 data/。"""
    configured = os.environ.get("CODEAGENT_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "data"


def default_db_path() -> Path:
    return default_data_root() / DEFAULT_DB_NAME


def build_parser() -> argparse.ArgumentParser:
    database = default_db_path()
    parser = argparse.ArgumentParser(description="索引本地 Python 代码库到 SQLite FTS5")
    parser.add_argument("--root", required=True, help="待索引仓库根目录")
    parser.add_argument("--name", default="", help="仓库名（默认取目录名）")
    parser.add_argument("--db", default=str(database), help=f"SQLite 路径（默认 {database}）")
    parser.add_argument("--reindex", action="store_true", help="清空该仓库旧索引后全量重建")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印 report")
    parser.add_argument(
        "--allow-root",
        action="append",
        default=None,
        help="允许索引的根目录，可重复；默认当前用户 home",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser()
    allowed = [Path(item).expanduser() for item in args.allow_root] if args.allow_root else None
    indexer = CodeIndexer(Path(args.db).expanduser(), allowed)
    try:
        report = indexer.index(str(root), args.name or root.name, reindex=bool(args.reindex))
    except (PermissionError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"repo: {report['repo_id']}")
        print(f"root: {root}")
        print(f"files: {report['files']}")
        print(f"symbols: {report['symbols']}")
        print(f"imports: {report['imports']}")
        print(f"references: {report['references']}")
        print(f"chunks: {report['chunks']}")
        print(f"skipped: {len(report['skipped'])}")
        for item in report["skipped"]:
            print(f"  - {item['path']}: {item['reason']}")
        print(f"errors: {len(report['errors'])}")
        for item in report["errors"]:
            print(f"  - {item['path']}: {item['error']}")
        print(f"ms: {report['ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
