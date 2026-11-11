"""REST API 路由：repos / search / symbol / impact / overview / ask / sessions。"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from ..agent.loop import AgentLimits, CodeAgent
from ..agent.tools import AgentToolError, AgentTools
from ..index.scanner import RootNotAllowedError
from ..retrieval.service import CodeService
from .errors import APIError, ServiceUnavailable
from .schemas import (
    AskRequest,
    RepoCreateRequest,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionListResponse,
)
from .services import AppServices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
DEFAULT_SEARCH_K = 8
MAX_SEARCH_K = 50
DEFAULT_SESSION_LIMIT = 50
MAX_SESSION_LIMIT = 200
MAX_QUERY_CHARS = 500


def get_services(request: Request) -> AppServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise ServiceUnavailable("服务尚未初始化")
    return services


ServicesDep = Annotated[AppServices, Depends(get_services)]


def _require(services: AppServices, name: str) -> Any:
    value = getattr(services, name, None)
    if value is None:
        detail = services.errors.get(name) or "依赖未初始化"
        raise ServiceUnavailable(f"{name} 不可用: {detail}")
    return value


def _repo_or_404(service: CodeService, repo_id: str) -> dict[str, Any]:
    repo = service.get_repo(repo_id)
    if repo is None:
        raise APIError(404, "repo_not_found", f"repo not found: {repo_id}")
    return repo


def _resolve_repo(service: CodeService, repo_id: str | None) -> dict[str, Any]:
    """未指定 repo_id 时：0 个仓库报 400，唯一仓库自动选中，多个仓库要求显式指定。"""
    if repo_id:
        return _repo_or_404(service, repo_id)
    repos = service.list_repos()
    if not repos:
        raise APIError(400, "no_repo_indexed", "没有已索引仓库，请先 POST /api/repos")
    if len(repos) > 1:
        raise APIError(400, "repo_id_required", "存在多个仓库，请指定 repo_id")
    return repos[0]


def _index_error(exc: Exception) -> APIError:
    if isinstance(exc, RootNotAllowedError):
        return APIError(403, "root_not_allowed", str(exc))
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
        return APIError(400, "invalid_root", str(exc))
    return APIError(400, "index_failed", f"{type(exc).__name__}: {exc}")


# ---------- repos ----------
@router.post("/repos", status_code=201)
async def create_repo(payload: RepoCreateRequest, services: ServicesDep) -> dict[str, Any]:
    indexer = _require(services, "indexer")
    service = _require(services, "service")
    try:
        report = await run_in_threadpool(
            indexer.index,
            payload.root,
            payload.name or "",
            payload.reindex,
        )
    except Exception as exc:  # noqa: BLE001 - 统一映射为业务错误
        raise _index_error(exc) from exc
    repo = service.get_repo(str(report.get("repo_id") or ""))
    return {"repo": repo, "report": report}


@router.get("/repos")
def list_repos(services: ServicesDep) -> dict[str, Any]:
    service = _require(services, "service")
    return {"repos": service.list_repos()}


@router.delete("/repos/{repo_id}")
def delete_repo(repo_id: str, services: ServicesDep) -> dict[str, Any]:
    service = _require(services, "service")
    if not service.delete_repo(repo_id):
        raise APIError(404, "repo_not_found", f"repo not found: {repo_id}")
    return {"deleted": True, "repo_id": repo_id}


@router.post("/repos/{repo_id}/reindex")
async def reindex_repo(repo_id: str, services: ServicesDep) -> dict[str, Any]:
    indexer = _require(services, "indexer")
    service = _require(services, "service")
    repo = _repo_or_404(service, repo_id)
    try:
        report = await run_in_threadpool(indexer.index, repo["root"], repo["name"], True)
    except Exception as exc:  # noqa: BLE001
        raise _index_error(exc) from exc
    return {"repo": service.get_repo(repo_id), "report": report}


# ---------- retrieval ----------
@router.get("/search")
def search(
    services: ServicesDep,
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARS)],
    k: Annotated[int, Query(ge=1, le=MAX_SEARCH_K)] = DEFAULT_SEARCH_K,
    repo_id: str | None = None,
    path_prefix: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    service = _require(services, "service")
    hits = service.search_code(q, k=k, repo_id=repo_id, path_prefix=path_prefix, kind=kind)
    return {"query": q, "count": len(hits), "hits": hits}


@router.get("/symbol")
def get_symbol(
    services: ServicesDep,
    qualname: Annotated[str, Query(min_length=1, max_length=500)],
    repo_id: Annotated[str, Query(min_length=1)],
    path: str | None = None,
) -> dict[str, Any]:
    service = _require(services, "service")
    symbol = service.get_symbol(repo_id, qualname, path=path)
    if symbol is None:
        raise APIError(404, "symbol_not_found", f"symbol not found: {qualname}")
    return {"symbol": symbol}


@router.get("/impact")
def impact(
    services: ServicesDep,
    repo_id: Annotated[str, Query(min_length=1)],
    path: Annotated[str, Query(min_length=1)],
) -> dict[str, Any]:
    service = _require(services, "service")
    return {"impact": service.impact(repo_id, path)}


@router.get("/overview")
def overview(
    services: ServicesDep,
    repo_id: Annotated[str, Query(min_length=1)],
) -> dict[str, Any]:
    service = _require(services, "service")
    return {"overview": service.overview(repo_id)}


# ---------- ask ----------
def _history_for(sessions: Any, session_id: str) -> list[dict[str, Any]]:
    if sessions is None:
        raise ServiceUnavailable("会话存储不可用")
    if sessions.get_session(session_id) is None:
        raise APIError(404, "session_not_found", f"session not found: {session_id}")
    return list(sessions.get_history(session_id))


def _persist_turn(sessions: Any, session_id: str, question: str, result: dict[str, Any]) -> None:
    answer = result.get("answer") or {}
    content = f"{answer.get('lead', '')}\n\n{answer.get('body', '')}".strip() or "（未能生成答案）"
    stats = result.get("stats") or {}
    meta = {
        "citations": result.get("citations") or [],
        "check": result.get("check") or "",
        "refusal": result.get("refusal"),
        "tool_calls": [str(item.get("name") or "") for item in result.get("tool_calls") or []],
        "turns": stats.get("turns"),
        "fallback": stats.get("fallback"),
    }
    sessions.append_message(session_id, "user", question)
    sessions.append_message(session_id, "assistant", content, meta=meta)


@router.post("/ask")
async def ask(payload: AskRequest, services: ServicesDep) -> dict[str, Any]:
    service = _require(services, "service")
    llm = _require(services, "llm")
    providers = getattr(llm, "providers", None)
    if providers is not None and len(providers) == 0:
        raise ServiceUnavailable("没有可用的 LLM provider（检查 LLM_PROVIDERS / API key）")

    repo = _resolve_repo(service, payload.repo_id)
    session_id = (payload.session_id or "").strip() or None
    history = _history_for(services.sessions, session_id) if session_id else []
    try:
        tools = AgentTools(service, repo["root"], repo["repo_id"])
    except AgentToolError as exc:
        raise APIError(400, "repo_unavailable", str(exc)) from exc

    factory = services.agent_factory
    if factory is None:
        limits = AgentLimits.from_settings(services.settings)
        factory = lambda bound: CodeAgent(bound, llm, limits)  # noqa: E731 - 保底装配，正式路径在 services
    agent = factory(tools)
    try:
        result = await run_in_threadpool(agent.ask, payload.question, history)
    except Exception as exc:  # noqa: BLE001 - Agent 崩溃统一 500
        logger.exception("agent failed")
        raise APIError(500, "agent_failed", f"{type(exc).__name__}: {exc}") from exc

    if session_id:
        try:
            _persist_turn(services.sessions, session_id, payload.question, result)
        except Exception as exc:  # noqa: BLE001 - 会话写失败不影响答案返回
            logger.warning("persist session turn failed: %s", exc)
    result["session_id"] = session_id
    result["repo_id"] = repo.get("repo_id")
    return result


# ---------- sessions ----------
@router.post("/sessions", status_code=201, response_model=SessionCreateResponse)
def create_session(payload: SessionCreateRequest, services: ServicesDep) -> dict[str, Any]:
    sessions = _require(services, "sessions")
    return {"session": sessions.create_session(payload.title)}


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    services: ServicesDep,
    limit: Annotated[int, Query(ge=1, le=MAX_SESSION_LIMIT)] = DEFAULT_SESSION_LIMIT,
) -> dict[str, Any]:
    sessions = _require(services, "sessions")
    return {"sessions": sessions.list_sessions(limit)}


__all__ = ["get_services", "router"]
