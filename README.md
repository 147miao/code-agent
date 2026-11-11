# code-agent

专门面向代码库的检索 Agent：索引本地仓库 → FTS5 检索 + 符号/影响面查询 → Agent 带 `path:line` 引用回答。
R2 在 SQLite 索引之上增加 Neo4j 代码图（AST 边 + LLM 候选边），用于更精确的 impact 查询。

## 状态
- P1：Python 代码索引（scanner/parser/chunker/SQLite FTS5/indexer/CLI）
- P2：Agent 工具循环 + API + 会话
- P3：Neo4j 代码图（AST 边 + LLM 候选边 + impact 回退 SQLite）
- P4：MySQL 会话适配

## 快速开始
```bash
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/index_code.py --root /path/to/repo --name myrepo --reindex
make api   # http://127.0.0.1:8000
```

## 代码图（R2）
```bash
docker compose up -d neo4j          # 7476:7474 / 7689:7687，数据在 ./.volumes/neo4j
make graph                          # 索引当前仓库 + 全量 AST 图；--with-llm 已启用

# 只对指定文件调用 LLM（AST 边仍全量重建）
python scripts/build_graph.py --root . --with-llm \
  --llm-path src/codeagent/index/chunker.py \
  --llm-path src/codeagent/citations.py
```

配置（`.env`）：`NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`，可选 `NEO4J_DATABASE`。
Neo4j 未配置或连接失败时：
- `POST /api/graph/build`、`GET /api/graph/status` 返回 503；
- `GET /api/impact` 自动回退 SQLite，响应 `impact.fallback="sqlite"`。

主要端点：
- `POST /api/graph/build` `{"repo_id": "...", "with_llm": false}`
- `GET /api/graph/status?repo_id=...`
- `GET /api/impact?repo_id=...&path=...&depth=1`（也支持 `qualname`，depth 1-2）

测试：
```bash
make test        # 默认跳过 integration
make test-graph  # 需要本机 7689 Neo4j；或设 NEO4J_TEST_URI
```

## 文档
- [plan-codeagent.md](plan-codeagent.md)
