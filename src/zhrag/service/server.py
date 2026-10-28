"""Single-process Uvicorn entry point with pre-drain investigation shutdown."""

from __future__ import annotations

import socket
from typing import Any

import uvicorn
from fastapi import FastAPI

from zhrag.investigation_executor import InvestigationExecutor


class InvestigationServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, executor: InvestigationExecutor | None = None):
        super().__init__(config)
        self.executor = executor

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        if self.executor:
            self.executor.begin_shutdown()
        await super().shutdown(sockets)


def create_server(app: FastAPI, **options: Any) -> InvestigationServer:
    # Bound other in-flight HTTP requests too; executor close has a separate
    # five-second total deadline for cooperative checkpoint completion.
    options.setdefault("timeout_graceful_shutdown", 5)
    return InvestigationServer(
        uvicorn.Config(app, **options), getattr(app.state, "investigations", None)
    )
