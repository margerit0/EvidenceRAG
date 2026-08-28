"""Optional HTTP service around the synchronous online retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zhrag.service.app import ServiceInfo

__all__ = ["ServiceInfo", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from zhrag.service import app  # noqa: PLC0415 - optional FastAPI boundary

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
