"""API 请求/响应模型（Pydantic v2，extra=forbid：未知字段直接 422）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_QUESTION_CHARS = 8000
MAX_TITLE_CHARS = 200
MAX_REPO_ID_CHARS = 64
MAX_ROOT_CHARS = 4096


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepoCreateRequest(StrictModel):
    root: str = Field(min_length=1, max_length=MAX_ROOT_CHARS, description="仓库根目录绝对路径")
    name: str | None = Field(default=None, max_length=200)
    reindex: bool = False


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    repo_id: str | None = Field(default=None, max_length=MAX_REPO_ID_CHARS)
    session_id: str | None = Field(default=None, max_length=MAX_REPO_ID_CHARS)


class GraphBuildRequest(StrictModel):
    repo_id: str = Field(min_length=1, max_length=MAX_REPO_ID_CHARS)
    with_llm: bool = False


class SessionCreateRequest(StrictModel):
    title: str = Field(default="", max_length=MAX_TITLE_CHARS)


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorBody


class SessionOut(StrictModel):
    id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""


class SessionCreateResponse(StrictModel):
    session: SessionOut


class SessionListResponse(StrictModel):
    sessions: list[SessionOut]
