"""集中配置入口：所有路径与连接串集中在这里，禁止在业务代码里硬编码。

环境变量与字段同名（无前缀），例如 DATA_ROOT / DB_PATH / SESSION_DB_PATH /
DATABASE_URL / LLM_PROVIDERS / AGENT_MAX_TOOL_CALLS / CODE_ALLOWED_ROOTS。
路径类字段为空字符串时按 data_root 下的约定目录解析（见 property）。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_SUBDIR = "data"
DEFAULT_DATA_ROOT = Path.home() / ".codeagent"
DEFAULT_LLM_CHAT_TIMEOUT = 45.0
DEFAULT_LLM_FIRST_TOKEN_TIMEOUT = 15.0

# 内置 LLM provider 链（与旧 RAG-M 保持一致，可用 LLM_PROVIDERS JSON 覆盖）。
# model 必须是对应服务实际返回的 id；api_key_env 只从 os.environ 读取，不写入仓库。
DEFAULT_LLM_PROVIDERS: tuple[dict[str, Any], ...] = (
    {
        "name": "opencode-go",
        "base_url": "http://127.0.0.1:8788/inference/go/v1",
        "model": "deepseek-v4.1-flash",
        "api_key_env": "OPENCODE_GO_API_KEY",
        # relay 要求每次请求带会话头；$UUID 由 LLMClient 每次请求替换为新的 uuid4。
        "extra_headers": {"x-opencode-session": "$UUID"},
        "first_token_timeout": 15,
    },
    {
        "name": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "gemma4:31b-cloud",
        "api_key": "ollama",
        "first_token_timeout": 20,
    },
    {
        "name": "ollama-qwen",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:1.5b-instruct",
        "api_key": "ollama",
        "first_token_timeout": 20,
    },
)


class Settings(BaseSettings):
    """code-agent 进程级配置。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- storage ----
    data_root: Path = DEFAULT_DATA_ROOT
    # 空字符串 = data_root/data/code_index.sqlite
    db_path: str = ""
    # 空字符串 = data_root/data/sessions.sqlite
    session_db_path: str = ""
    # 空字符串 = SQLite；mysql+pymysql://user:pass@host:port/db?charset=utf8mb4 时用 MySQL
    database_url: str = ""

    # ---- llm ----
    # JSON 数组；空字符串 = DEFAULT_LLM_PROVIDERS
    llm_providers: str = ""
    llm_chat_timeout: float = DEFAULT_LLM_CHAT_TIMEOUT
    llm_first_token_timeout: float = DEFAULT_LLM_FIRST_TOKEN_TIMEOUT
    llm_temperature: float = 0.0

    # ---- agent 预算 ----
    agent_max_tool_calls: int = 6
    agent_max_turns: int = 8
    agent_timeout_s: float = 120.0

    # ---- 允许索引/读取的根目录（默认仅当前用户 home）----
    code_allowed_roots: list[str] = Field(default_factory=lambda: [str(Path.home())])

    @property
    def data_dir(self) -> Path:
        return Path(self.data_root).expanduser() / DATA_SUBDIR

    @property
    def code_store_path(self) -> Path:
        """代码索引 SQLite 路径；DB_PATH 为空时落到 data_root/data/code_index.sqlite。"""
        if self.db_path.strip():
            return Path(self.db_path).expanduser()
        return self.data_dir / "code_index.sqlite"

    @property
    def session_store_path(self) -> Path:
        """会话 SQLite 路径；SESSION_DB_PATH 为空时落到 data_root/data/sessions.sqlite。"""
        if self.session_db_path.strip():
            return Path(self.session_db_path).expanduser()
        return self.data_dir / "sessions.sqlite"

    @property
    def allowed_roots(self) -> list[Path]:
        return [Path(item).expanduser() for item in self.code_allowed_roots]

    @property
    def use_mysql(self) -> bool:
        return bool(self.database_url.strip())

    @property
    def resolved_llm_providers(self) -> list[dict[str, Any]]:
        """解析 LLM_PROVIDERS；空字符串返回内置 provider 链。"""
        raw = self.llm_providers.strip()
        if not raw:
            return [dict(item) for item in DEFAULT_LLM_PROVIDERS]
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise TypeError("LLM_PROVIDERS 必须是 JSON 数组")
        return [dict(item) for item in parsed]

    def ensure_dirs(self) -> None:
        """创建索引/会话目录（SQLite 模式）；MySQL 模式只创建数据目录。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.use_mysql:
            self.session_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.code_store_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
