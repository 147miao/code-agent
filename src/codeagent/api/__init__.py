"""API 层：FastAPI 装配与路由。"""

from __future__ import annotations

from .main import create_app
from .services import AppServices, build_services

__all__ = ["AppServices", "build_services", "create_app"]
