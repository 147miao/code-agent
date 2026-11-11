"""API 统一错误响应：{error: {code, message, request_id}}。

422 参数校验沿用 FastAPI 默认（不包装）；业务错误由 APIError 或领域异常映射：
400 参数/请求不合法、403 路径越权、404 资源不存在、503 依赖不可用。
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..index.scanner import RootNotAllowedError
from ..retrieval.service import CodeRepoNotFound, CodeValidationError
from ..sessions.repository import SessionNotFoundError

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class APIError(Exception):
    """带 HTTP 状态与业务 code 的异常。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)


class ServiceUnavailable(APIError):
    def __init__(self, message: str, *, code: str = "service_unavailable") -> None:
        super().__init__(503, code, message)


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "") or uuid.uuid4().hex[:12])


def error_body(request: Request, code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message, "request_id": request_id(request)}}


def _json(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(request, code, message))


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return _json(request, exc.status_code, exc.code, exc.message)


async def repo_not_found_handler(request: Request, exc: CodeRepoNotFound) -> JSONResponse:
    return _json(request, 404, "repo_not_found", str(exc))


async def validation_error_handler(request: Request, exc: CodeValidationError) -> JSONResponse:
    return _json(request, 400, "invalid_argument", str(exc))


async def root_not_allowed_handler(request: Request, exc: RootNotAllowedError) -> JSONResponse:
    return _json(request, 403, "root_not_allowed", str(exc))


async def missing_path_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return _json(request, 400, "invalid_root", str(exc))


async def not_a_directory_handler(request: Request, exc: NotADirectoryError) -> JSONResponse:
    return _json(request, 400, "invalid_root", str(exc))


async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    return _json(request, 404, "session_not_found", str(exc))


async def request_id_middleware(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    return response


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(CodeRepoNotFound, repo_not_found_handler)
    app.add_exception_handler(CodeValidationError, validation_error_handler)
    app.add_exception_handler(RootNotAllowedError, root_not_allowed_handler)
    app.add_exception_handler(FileNotFoundError, missing_path_handler)
    app.add_exception_handler(NotADirectoryError, not_a_directory_handler)
    app.add_exception_handler(SessionNotFoundError, session_not_found_handler)


__all__ = [
    "REQUEST_ID_HEADER",
    "APIError",
    "ServiceUnavailable",
    "error_body",
    "install_error_handlers",
    "request_id",
    "request_id_middleware",
]
