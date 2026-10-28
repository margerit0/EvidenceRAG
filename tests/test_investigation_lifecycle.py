"""Real local TCP lifecycle, with original synthetic dependencies only."""

from __future__ import annotations

import asyncio
import threading

import pytest

from zhrag.investigation_executor import BoundaryExecution
from zhrag.service.investigation_demo import create_demo_app
from zhrag.service.server import create_server


@pytest.mark.parametrize("status", ["paused", "waiting_input"])
def test_graceful_shutdown_drains_open_sse_and_closes_executor(tmp_path, status):
    async def scenario():
        app = create_demo_app(tmp_path / "shutdown.sqlite3", delay=0)
        executor = app.state.investigations
        run_id = executor.create("shutdown-create", "原创合成问题", "synthetic-demo")
        if status == "paused":
            executor.store.command(run_id, "pause-key", "pause", version=1)
        else:
            agent = executor.models["synthetic-demo"]
            epoch = executor.store.claim(
                run_id, "setup", agent.profile_fingerprint, executor.snapshot
            )
            BoundaryExecution(
                executor.store,
                agent,
                run_id,
                owner="setup",
                epoch=epoch,
                shutdown=threading.Event(),
            ).execute()
        assert executor.store.get(run_id)["status"] == status
        server = create_server(app, host="127.0.0.1", port=0, log_level="error")
        serving = asyncio.create_task(server.serve())
        writer = None
        try:

            async def ready():
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(ready(), 3)
            port = server.servers[0].sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                (
                    f"GET /api/investigations/{run_id}/stream HTTP/1.1\r\nHost: localhost\r\n\r\n"
                ).encode()
            )
            await writer.drain()
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
            assert b"200 OK" in headers and b"text/event-stream" in headers
            # This is Uvicorn's first graceful-exit path; the client stays open.
            server.should_exit = True
            await asyncio.wait_for(asyncio.shield(serving), 2)
            assert executor.stop.is_set() and not executor.thread.is_alive()
            assert not server.force_exit and not server.server_state.tasks
            assert executor.store.get(run_id)["status"] == status
            assert (await asyncio.wait_for(reader.read(), 1)).endswith(b"0\r\n\r\n")
        finally:
            if writer:
                writer.close()
                await writer.wait_closed()
            server.should_exit = True
            await asyncio.wait_for(serving, 6)

    asyncio.run(scenario())
