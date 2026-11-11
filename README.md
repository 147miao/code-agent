# code-agent

专门面向代码库的检索 Agent：索引本地仓库 → FTS5 检索 + 符号/影响面查询 → Agent 带 `path:line` 引用回答。

## 状态
- P1：Python 代码索引（scanner/parser/chunker/SQLite FTS5/indexer/CLI）
- P2：Agent 工具循环 + API + 会话
- P3：Neo4j 代码图（AST 边 + LLM 候选边）
- P4：MySQL 会话适配

## 快速开始
```bash
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/index_code.py --root /path/to/repo --name myrepo --reindex
make api   # 待 P2
```

## 文档
- [plan-codeagent.md](plan-codeagent.md)