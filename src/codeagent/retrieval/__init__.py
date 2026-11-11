"""代码库检索服务层。

- service：repo / search / symbol / overview / impact 的门面（ask 由 Agent 工作流负责）。
"""
from __future__ import annotations

from .service import (
    DEFAULT_SEARCH_K,
    MAX_SEARCH_K,
    CodeRepoNotFound,
    CodeService,
    CodeServiceError,
    CodeStoreProtocol,
    CodeValidationError,
)

__all__ = [
    "DEFAULT_SEARCH_K",
    "MAX_SEARCH_K",
    "CodeRepoNotFound",
    "CodeService",
    "CodeServiceError",
    "CodeStoreProtocol",
    "CodeValidationError",
]
