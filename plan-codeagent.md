# code-agent 计划 v2（吸收独立审核）

## 0. 需求与范围（用户确认）
- 只保留代码库检索 + 专用 agent；新仓库/新包；旧 RAG-M 已停；
- Python first；Neo4j 代码图（AST 边 + opencode-go deepseek 候选边）；会话 MySQL；允许重构。
- **分两轮验收**：R1 = 索引/检索 + Agent/API/会话（含 MySQL 适配器）；R2 = Neo4j 代码图。

## 1. 验收标准（可证伪）
| # | 标准 | 测法 |
|---|---|---|
| A1 | 扫描/解析：Python AST；qualname/行号/imports/references；排除 .venv*/site-packages/data；allowed_roots + 符号链接拒绝 | 临时仓库单测 |
| A2 | 索引：FTS5 **unicode61 + trigram 双表 RRF**；file_hash/chunk_id 确定性；重复索引幂等（计数与 chunk_id 不变）；改/删文件清理；语法错误重试 | 单测断言 |
| A3 | 检索：10 题 fixtures hit@5 = **10/10**（含 ≥3 道纯中文）；search < 100ms（1k 文件） | tests + 计时 |
| A4 | Agent：手写循环（无 LangGraph）；工具 search_code/get_symbol/read_file/list_dir/grep/impact/overview；预算 = tool calls ≤6、LLM turns ≤8、超时 120s；read_file/list_dir/grep 限定 repo root（realpath，禁符号链接逃逸）并限行/条数 | fake LLM + 单测 |
| A5 | 会话：SessionRepository 契约；SQLite 默认（hermetic）+ MySQL 适配器（compose mysql:8.4；缺 DSN 时集成测试 skip 并标未验证）；ask 支持 session_id | 仓库契约测试 + 集成标记 |
| A6 | API：/api/repos、/search、/symbol、/impact、/overview、/ask、/sessions；**参数错误统一 422（FastAPI 默认）**、业务错误 400/403/404/503；错误体一致 | TestClient 全端点 |
| A7 | 自举 code-agent 自身：10 题（路径+行号 ground truth），非空检索必须带有效引用（编号连续、行号 ≤EOF、path 在库），refusal=0；伪造 [99]/越权/无证据有测试 | scripts/code_e2e.py + 单测 |
| A8 | 测试/CI：pytest + ruff + make ci 全绿；**导入 codeagent 后无 torch/transformers/Milvus**（子进程断言）；1k 文件索引 < 60s | CI |
| A9 | R2 Neo4j：schema/复合键/索引；AST 边（CONTAINS/IMPORTS/REFERENCES）+ LLM 候选边（CALLS/TESTS，source/confidence/evidence）；过滤 builtins/self/短名与 fanout 上限；LLM 边按 file_hash 缓存、token 预算、阈值；Neo4j 不可用 impact 回退 SQLite；make test-graph（docker/testcontainers，CI 默认 skip） | R2 验收 |
| A10 | 文档/CLI：README、docs/architecture.md、.env.example；CLI 可独立跑；uv.lock | diff + 实跑 |

## 2. 结构（冻结）
```
src/codeagent/
  config.py
  citations.py            # parse_structured_answer / parse_citations / verify_code_citations（本地，无 rag 依赖）
  index/                  # scanner, parser, chunker, store, indexer
  retrieval/              # service: search/get_symbol/overview/impact
  agent/                  # loop.py, tools.py, prompts.py（手写 while 循环）
  api/                    # main, routes, schemas
  llm/                    # client.py, router.py（provider 链，复制精简）
  sessions/               # repository.py(sqlite), mysql.py, models
  codegraph/              # R2: neo4j schema/import/edges（不叫 graph，避免与 agent 混淆）
scripts/index_code.py
tests/
```

### 契约
```python
class CodeStore:
    add_repo/list_repos/get_repo/delete_repo/get_file_hash/upsert_file/delete_file_rows
    insert_symbols/insert_imports/insert_references/insert_chunks; search; get_symbol; overview; reverse_imports
class SessionRepository:
    create_session/get_session/list_sessions/delete_session/append_message/get_history
class AgentTools:
    search_code(question,k) / get_symbol(qualname,path) / read_file(path,start?,end?) / list_dir(path)
    grep(pattern,path_prefix?) / impact(path) / overview()   # 全部限定 repo root
```

## 3. 设计要点（审核修正）
- **无重型依赖**：citations 本地化；LLM 只复制 llm.py/router.py；导入测试断言无 torch；
- **中文检索**：`chunks_fts`（unicode61）+ `chunks_fts_trigram`（trigram）双表，RRF 合并；A3 含中文题；
- **Agent 引用**：只允许本轮工具返回的 contexts 编号；校验 path 在库、行号 ≤ EOF、编号连续；无效引用剔除；无有效引用→拒答或抽取式兜底；
- **工具安全**：realpath 必须位于 repo root；禁 symlink；read_file ≤200 行/次、grep ≤100 条、list_dir ≤200 项；
- **API**：422 为参数校验（FastAPI 默认），业务错误 400/403/404/503；统一 `{error:{code,message,request_id}}`；
- **MySQL**：SessionRepository 两实现；compose `mysql:8.4`（3307 映射避免占 3306），DSN 从 `DATABASE_URL`；集成测试标记 `@pytest.mark.integration`，CI 默认 skip；
- **R2 Neo4j**：节点键 `(repo_id,kind,qualname)`；索引；AST 边 source=ast confidence=1.0；LLM 边候选集≤30、阈值 0.6、`file_hash` 缓存、token 预算；fanout 上限 50；reindex/delete 同步；impact 回退 SQLite。

## 4. 执行
| 工作流 | Owner | 内容 |
|---|---|---|
| W1 索引/检索 | 子 agent A | index/* + retrieval/* + citations.py + CLI + 测试（移植旧 code，去 torch/文档耦合） |
| W2 Agent/API/会话 | 子 agent B | agent/* + api/* + sessions/*(SQLite+MySQL 适配器) + llm/* + 测试（用 fake store） |
| W3 Neo4j 图 | R2 子 agent C | codegraph/* + LLM 候选边 + 测试（docker 标记） |
| 集成/文档 | 主 agent | 合并、main 装配、自举 E2E、README/docs、CI |

## 5. 依赖/环境
- pyproject：fastapi/uvicorn/pydantic-settings/httpx/python-dotenv/orjson/pymysql[rsa]/neo4j；**不引入 torch/langgraph**；dev：pytest/pytest-asyncio/ruff；
- .venv 用 uv 创建；uv.lock 提交；GitHub Actions：Python3.13 + ruff + pytest + FTS5 检查；integration/graph 标记默认跳过。

## 6. 不在本轮
TS/Go（tree-sitter）、前端 UI、dense 向量、MCP、多用户鉴权。
