"""Neo4j 惰性客户端：驱动只在首次使用时创建，配置缺失/连接失败统一抛 GraphUnavailableError。

设计：
- 不 import 时连接：Neo4j 不可用不能阻塞 API 启动；
- verify_connectivity 失败时关闭驱动并缓存 None，下次调用重试；
- 查询异常会重置驱动，下一次调用重新建连；
- 驱动工厂可注入（测试用假驱动，不依赖真实 Neo4j）。
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0


class GraphError(RuntimeError):
    """代码图相关错误基类。"""


class GraphUnavailableError(GraphError):
    """Neo4j 未配置、连接失败或查询失败。"""


class GraphValidationError(GraphError, ValueError):
    """图查询/构建参数不合法。"""


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j 连接配置（与 Settings 解耦，便于测试与脚本复用）。"""

    uri: str
    user: str
    password: str
    database: str = ""
    timeout: float = DEFAULT_TIMEOUT

    @property
    def configured(self) -> bool:
        return bool(self.uri.strip() and self.user.strip() and self.password)

    @classmethod
    def from_settings(cls, settings: Any) -> Neo4jConfig:
        return cls(
            uri=str(getattr(settings, "neo4j_uri", "") or "").strip(),
            user=str(getattr(settings, "neo4j_user", "") or "").strip(),
            password=str(getattr(settings, "neo4j_password", "") or ""),
            database=str(getattr(settings, "neo4j_database", "") or "").strip(),
            timeout=float(getattr(settings, "neo4j_timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT),
        )

    def describe(self) -> dict[str, Any]:
        """可安全返回给 API 的配置摘要（绝不包含密码）。"""
        return {
            "uri": self.uri,
            "database": self.database or "default",
            "configured": self.configured,
        }


class Neo4jClient:
    """线程安全的惰性 Neo4j 客户端。"""

    def __init__(
        self,
        config: Neo4jConfig,
        *,
        driver_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._driver_factory = driver_factory
        self._driver: Any | None = None
        # RLock：driver 属性持锁建连，失败路径会 reset()，避免自锁。
        self._lock = threading.RLock()

    @property
    def configured(self) -> bool:
        return self.config.configured

    @property
    def driver(self) -> Any:
        """首次访问时建连并 verify_connectivity；失败抛 GraphUnavailableError。"""
        if not self.configured:
            raise GraphUnavailableError("Neo4j 未配置（NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD）")
        with self._lock:
            if self._driver is None:
                self._driver = self._connect()
            return self._driver

    def _connect(self) -> Any:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - pyproject 已声明 neo4j
            raise GraphUnavailableError("未安装 neo4j Python 驱动") from exc
        factory = self._driver_factory or GraphDatabase.driver
        driver: Any | None = None
        try:
            driver = factory(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                connection_timeout=self.config.timeout,
                max_connection_lifetime=300,
            )
            driver.verify_connectivity()
        except Exception as exc:
            if driver is not None:
                try:
                    driver.close()
                except Exception:  # noqa: BLE001 - 连接失败后的清理不应掩盖根因
                    logger.debug("neo4j driver close failed", exc_info=True)
            raise GraphUnavailableError(f"Neo4j 连接失败: {type(exc).__name__}: {exc}") from exc
        logger.info("neo4j connected: %s database=%s", self.config.uri, self.config.database or "default")
        return driver

    def reset(self) -> None:
        with self._lock:
            driver, self._driver = self._driver, None
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响后续重连
                logger.debug("neo4j driver close failed", exc_info=True)

    def close(self) -> None:
        self.reset()

    def records(self, query: str, parameters: dict[str, Any] | None = None) -> list[Any]:
        """执行 Cypher 并返回原始 Record 列表（保留节点/关系对象，供图查询解析）。"""
        driver = self.driver
        database = self.config.database or None
        try:
            with driver.session(database=database) as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Exception as exc:
            self.reset()
            raise GraphUnavailableError(f"Neo4j 查询失败: {type(exc).__name__}: {exc}") from exc

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """执行 Cypher 并把每行转成普通 dict（计数/DDL 等无节点对象场景）。"""
        return [record.data() for record in self.records(query, parameters)]


__all__ = [
    "DEFAULT_TIMEOUT",
    "GraphError",
    "GraphUnavailableError",
    "GraphValidationError",
    "Neo4jClient",
    "Neo4jConfig",
]
