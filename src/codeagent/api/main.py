"""FastAPI 应用装配：lifespan 构建 CodeStore/CodeIndexer/CodeService/Agent 依赖 + 会话仓储。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from ..config import Settings, get_settings
from .errors import install_error_handlers, request_id_middleware
from .routes import router
from .services import AppServices, build_services

logger = logging.getLogger(__name__)


APP_TITLE = "code-agent"
APP_VERSION = "0.1.0"
API_PREFIX = "/api"


def _lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = build_services(settings)
        yield

    return lifespan


def create_app(settings: Settings | None = None, *, services: AppServices | None = None) -> FastAPI:
    """构建应用。

    services 显式传入时不注册 lifespan（测试/嵌入式场景直接使用装配好的依赖）；
    否则在 startup 时构建，失败依赖置 None，对应端点返回 503。
    """
    resolved = settings or get_settings()
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        lifespan=None if services is not None else _lifespan(resolved),
    )
    app.state.services = services or AppServices(settings=resolved, errors={"startup": "not_started"})
    app.middleware("http")(request_id_middleware)
    install_error_handlers(app)
    app.include_router(router)

    @app.get(f"{API_PREFIX}/health")
    def health(request: Request) -> dict[str, object]:
        current: AppServices = getattr(request.app.state, "services", None) or AppServices(settings=resolved)
        llm = current.llm
        providers = llm.provider_status() if callable(getattr(llm, "provider_status", None)) else {}
        graph = getattr(current, "graph", None)
        describe = getattr(graph, "describe", None)
        return {
            "status": "ok" if not current.errors else "degraded",
            "deps": current.availability(),
            "errors": current.errors,
            "providers": providers,
            # 只展示连接配置，不触发 Neo4j 连接。
            "graph": describe() if callable(describe) else {"configured": False},
        }

    return app


app = create_app()

__all__ = ["app", "create_app"]
