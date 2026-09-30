"""FastAPI application factory with no provider or database composition policy.

The online retriever is deliberately synchronous: its embedding, Milvus, and
rerank clients are blocking. The HTTP boundary runs it in Starlette's threadpool
and admits only a bounded number of requests, so one slow provider cannot turn
Milvus Lite's single local database into an unbounded work queue.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from zhrag.agent import AGENT_MESSAGES, DocumentAgent
from zhrag.agent_progress import AgentProgress
from zhrag.answering import PUBLIC_MESSAGES, Answerer, AnswerOutcome, AnswerStatus
from zhrag.retrieval.online import OnlineRetrievalResult, OnlineRetriever, StageTimings
from zhrag.service.agent_stream import ProgressFeed
from zhrag.service.observability import TraceRecorder, TraceSink, profile_fingerprint
from zhrag.store.base import MetadataValue

__all__ = ["ServiceInfo", "create_app"]

QUERY_MAX_LENGTH = 2_000
DISPLAY_TEXT_MAX_LENGTH = 4_000
_ALLOWED_METADATA = frozenset(
    {
        "approx_tokens",
        "category",
        "collection",
        "heading_path",
        "path",
        "source_url",
        "theme",
    }
)


@dataclass(frozen=True, slots=True)
class ServiceInfo:
    """Non-secret deployment metadata exposed by the health endpoint."""

    profile_name: str
    embedding_profile: str
    rerank_profile: str
    rerank_enabled: bool

    def __post_init__(self) -> None:
        for name in ("profile_name", "embedding_profile", "rerank_profile"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")


class SearchRequest(BaseModel):
    """The only accepted request fields for one retrieval."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: Annotated[str, Field(min_length=1, max_length=QUERY_MAX_LENGTH)]
    top_k: Annotated[int, Field(strict=True, ge=1)] | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    profile_name: str
    embedding_profile: str
    rerank_profile: str
    rerank_enabled: bool
    max_concurrency: int


class TimingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_encode_seconds: float
    sparse_encode_seconds: float
    dense_search_seconds: float
    sparse_search_seconds: float
    fusion_seconds: float
    fetch_seconds: float
    rerank_seconds: float
    total_seconds: float


class PassageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    fused_rank: int
    rerank_score: float
    doc_id: str
    source_key: str
    ordinal: int
    metadata: dict[str, MetadataValue]
    text: str
    text_truncated: bool


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_name: str
    embedding_profile: str
    rerank_profile: str
    rerank_enabled: bool
    passages: list[PassageResponse]
    timings: TimingResponse


class CapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_enabled: bool
    output_limit: int
    generation_profile: str | None
    agent_enabled: bool = False
    agent_profile: str | None = None
    agent_streaming: bool = False


class InvestigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: Annotated[str, Field(min_length=1, max_length=QUERY_MAX_LENGTH)]


class AnswerBlockResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[int]


class AnswerSourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: int
    rank: int
    title: str
    text: str
    source_url: str | None


class ContextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_count: int
    skipped_budget_count: int
    prompt_estimated_tokens: int


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    message: str
    blocks: list[AnswerBlockResponse]
    sources: list[AnswerSourceResponse]
    retrieval: SearchResponse
    generation_profile: str
    context: ContextResponse
    generation_seconds: float
    total_seconds: float


class _RetrieverLike(Protocol):
    settings: object

    def retrieve(self, query: str) -> OnlineRetrievalResult: ...


SearchRunner = Callable[[str], OnlineRetrievalResult | Awaitable[OnlineRetrievalResult]]


@dataclass(slots=True)
class _AdmissionGate:
    """Atomic fail-fast admission around a bounded number of active requests."""

    limit: int
    _active: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active < 1:
                raise RuntimeError("admission gate released without an active request")
            self._active -= 1


def _no_store_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store"}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(code=code, message=message).model_dump(),
        headers=_no_store_headers(),
    )


def _metadata_allowlist(
    metadata: Mapping[str, MetadataValue],
) -> dict[str, MetadataValue]:
    return {key: metadata[key] for key in sorted(_ALLOWED_METADATA & metadata.keys())}


def _timing_values(timings: StageTimings) -> dict[str, float]:
    values = {name: float(getattr(timings, name)) for name in TimingResponse.model_fields}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise RuntimeError("retriever returned invalid stage timings")
    return values


def _timings_response(timings: StageTimings) -> TimingResponse:
    return TimingResponse(**_timing_values(timings))


def _search_response(
    result: OnlineRetrievalResult,
    *,
    info: ServiceInfo,
    top_k: int,
) -> SearchResponse:
    if (
        result.profile_name != info.profile_name
        or result.embedding_profile != info.embedding_profile
        or result.rerank_profile != info.rerank_profile
    ):
        raise RuntimeError("retriever profile changed after service composition")
    rows: list[PassageResponse] = []
    for row in result.passages[:top_k]:
        passage = row.passage
        text = passage.text[:DISPLAY_TEXT_MAX_LENGTH]
        rows.append(
            PassageResponse(
                rank=row.rank,
                fused_rank=row.fused_rank,
                rerank_score=row.rerank_score,
                doc_id=passage.doc_id,
                source_key=passage.source_key,
                ordinal=passage.ordinal,
                metadata=_metadata_allowlist(passage.metadata),
                text=text,
                text_truncated=len(text) < len(passage.text),
            )
        )
    return SearchResponse(
        profile_name=result.profile_name,
        embedding_profile=result.embedding_profile,
        rerank_profile=result.rerank_profile,
        rerank_enabled=info.rerank_enabled,
        passages=rows,
        timings=_timings_response(result.timings),
    )


def _default_info(retriever: OnlineRetriever) -> ServiceInfo:
    settings = retriever.settings
    return ServiceInfo(
        profile_name=settings.profile_name,
        embedding_profile=settings.embedding_profile,
        rerank_profile=settings.rerank_profile,
        rerank_enabled=settings.rerank_enabled,
    )


def _trace_retrieval_settings(settings: object) -> dict[str, object]:
    names = (
        "dense_dimensions",
        "arm_depth",
        "fusion_depth",
        "rrf_k",
        "rerank_request_depth",
        "rerank_apply_depth",
        "output_limit",
    )
    return {name: getattr(settings, name) for name in names}


def _safe_source_url(value: MetadataValue) -> str | None:
    if not isinstance(value, str) or len(value) > 4_000:
        return None
    if any(ord(char) < 33 for char in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
        ):
            return value
    except ValueError:
        pass
    return None


def _ask_response(
    outcome: AnswerOutcome,
    retrieval: SearchResponse,
    *,
    total_seconds: float,
) -> AskResponse:
    sources = [
        AnswerSourceResponse(
            citation_id=row.citation_id,
            rank=row.ranked.rank,
            title=row.title,
            text=row.ranked.passage.text,
            source_url=_safe_source_url(row.ranked.passage.metadata.get("source_url")),
        )
        for row in outcome.context.evidence
    ]
    return AskResponse(
        status=outcome.status,
        message=PUBLIC_MESSAGES[outcome.status],
        blocks=[
            AnswerBlockResponse(text=b.text, citations=list(b.citations)) for b in outcome.blocks
        ],
        sources=sources,
        retrieval=retrieval,
        generation_profile=outcome.profile_fingerprint,
        context=ContextResponse(
            evidence_count=len(sources),
            skipped_budget_count=outcome.context.skipped_budget_count,
            prompt_estimated_tokens=outcome.context.prompt_estimated_tokens,
        ),
        generation_seconds=outcome.generation_seconds,
        total_seconds=total_seconds,
    )


def create_app(  # noqa: PLR0915
    retriever: OnlineRetriever,
    *,
    info: ServiceInfo | None = None,
    static_dir: Path | None = None,
    frontend_dir: Path | None = None,
    max_concurrency: int = 1,
    search_runner: SearchRunner | None = None,
    answerer: Answerer | None = None,
    agent: DocumentAgent | None = None,
    trace_sink: TraceSink | None = None,
    published_index_identity: str | None = None,
) -> FastAPI:
    """Create an app around an already-composed retriever.

    Importing this module and calling this factory neither reads ``.env`` nor
    opens Milvus. The composition root owns those side effects and may inject an
    async runner in tests; production uses the threadpool-backed default.
    """
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise TypeError("max_concurrency must be an integer")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    service_info = info or _default_info(retriever)
    settings = retriever.settings
    if (
        service_info.profile_name != settings.profile_name
        or service_info.embedding_profile != settings.embedding_profile
        or service_info.rerank_profile != settings.rerank_profile
        or service_info.rerank_enabled != settings.rerank_enabled
    ):
        raise ValueError("service info does not match retriever settings")
    output_limit = settings.output_limit
    if agent is not None and agent.retriever is not retriever:
        raise ValueError("agent must share the service retriever and admission gate")
    resolved_static = static_dir or Path(__file__).resolve().parent / "static"
    index_path = resolved_static / "index.html"
    runner = search_runner
    recorder = TraceRecorder(
        trace_sink,
        profile_name=service_info.profile_name,
        profile_fingerprint_value=profile_fingerprint(
            profile_name=service_info.profile_name,
            embedding_profile=service_info.embedding_profile,
            rerank_profile=service_info.rerank_profile,
            rerank_enabled=service_info.rerank_enabled,
            output_limit=output_limit,
            retrieval_settings=_trace_retrieval_settings(settings),
            published_index_identity=published_index_identity,
        ),
        rerank_enabled=service_info.rerank_enabled,
    )

    app = FastAPI(
        title="zhrag TiDB retrieval service",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.retriever = retriever
    app.state.service_info = service_info
    app.state.max_concurrency = max_concurrency
    app.state.admission = _AdmissionGate(max_concurrency)
    # Keep shielded work alive when its HTTP caller disconnects.
    pending_asks: set[asyncio.Task[AskResponse | JSONResponse]] = set()
    pending_investigations: set[asyncio.Task[JSONResponse]] = set()

    @app.get("/api/capabilities", response_model=CapabilitiesResponse)
    async def capabilities(response: Response) -> CapabilitiesResponse:
        response.headers.update(_no_store_headers())
        return CapabilitiesResponse(
            generation_enabled=answerer is not None,
            output_limit=output_limit,
            generation_profile=answerer.profile_fingerprint if answerer else None,
            agent_enabled=agent is not None,
            agent_profile=agent.profile_fingerprint if agent else None,
            agent_streaming=agent is not None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        span = recorder.start_span()
        span.end(status_code=422, outcome="invalid_request")
        return _error(422, "invalid_request", "The request body is invalid.")

    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> FileResponse | JSONResponse:
        if not index_path.is_file():
            return _error(404, "static_unavailable", "The web interface is unavailable.")
        return FileResponse(index_path, headers=_no_store_headers())

    @app.get("/healthz", response_model=HealthResponse)
    async def health(response: Response) -> HealthResponse:
        response.headers.update(_no_store_headers())
        return HealthResponse(
            status="ok",
            profile_name=service_info.profile_name,
            embedding_profile=service_info.embedding_profile,
            rerank_profile=service_info.rerank_profile,
            rerank_enabled=service_info.rerank_enabled,
            max_concurrency=max_concurrency,
        )

    @app.post(
        "/api/search",
        response_model=SearchResponse,
        responses={
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def search(
        payload: SearchRequest,
        response: Response,
    ) -> SearchResponse | JSONResponse:
        response.headers.update(_no_store_headers())
        top_k = payload.top_k or output_limit
        if top_k > output_limit:
            span = recorder.start_span()
            span.end(status_code=422, outcome="invalid_top_k")
            return _error(
                422,
                "invalid_top_k",
                f"top_k must be between 1 and {output_limit}.",
            )
        admission = app.state.admission
        if not admission.try_acquire():
            span = recorder.start_span()
            span.end(status_code=429, outcome="admission_rejected")
            return _error(429, "service_busy", "The service is at its concurrency limit.")
        span = recorder.start_span()
        try:
            try:
                if runner is None:
                    result = await run_in_threadpool(retriever.retrieve, payload.query)
                else:
                    value = runner(payload.query)
                    result = await value if inspect.isawaitable(value) else value
                output = _search_response(result, info=service_info, top_k=top_k)
                span.end(
                    status_code=200,
                    outcome="success",
                    candidate_count=len(result.fused_candidates),
                    output_count=len(output.passages),
                    stage_timings=_timing_values(result.timings),
                )
                return output
            except Exception:
                span.end(status_code=503, outcome="retrieval_failed")
                return _error(
                    503,
                    "retrieval_failed",
                    "Retrieval failed; try again later.",
                )
        finally:
            admission.release()

    async def run_ask(
        payload: SearchRequest,
        top_k: int,
        active_answerer: Answerer,
    ) -> AskResponse | JSONResponse:
        started = time.perf_counter()
        span = recorder.start_span()
        try:
            try:
                if runner is None:
                    result = await run_in_threadpool(retriever.retrieve, payload.query)
                else:
                    value = runner(payload.query)
                    result = await value if inspect.isawaitable(value) else value
                retrieval = _search_response(result, info=service_info, top_k=top_k)
            except (Exception, SystemExit):
                span.end(status_code=503, outcome="retrieval_failed")
                return _error(503, "retrieval_failed", "Retrieval failed; try again later.")
            # M11 remains a retrieval-only span, not an end-to-end generation trace.
            span.end(
                status_code=200,
                outcome="success",
                candidate_count=len(result.fused_candidates),
                output_count=len(retrieval.passages),
                stage_timings=_timing_values(result.timings),
            )
            try:
                outcome = await run_in_threadpool(
                    active_answerer.answer,
                    payload.query,
                    result.passages[:top_k],
                )
                output = _ask_response(
                    outcome,
                    retrieval,
                    total_seconds=time.perf_counter() - started,
                )
            except (Exception, SystemExit):
                return JSONResponse(
                    status_code=503,
                    content={
                        "code": "generation_failed",
                        "message": PUBLIC_MESSAGES["generation_failed"],
                        "retrieval": retrieval.model_dump(),
                    },
                    headers=_no_store_headers(),
                )
            if outcome.status in {"generation_failed", "generation_timeout", "invalid_answer"}:
                return JSONResponse(
                    status_code=503,
                    content=output.model_dump(),
                    headers=_no_store_headers(),
                )
            return output
        finally:
            app.state.admission.release()

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(payload: SearchRequest, response: Response) -> AskResponse | JSONResponse:
        response.headers.update(_no_store_headers())
        if answerer is None:
            return _error(503, "generation_unavailable", "Answer generation is not enabled.")
        top_k = payload.top_k or output_limit
        if top_k > output_limit:
            return _error(422, "invalid_top_k", f"top_k must be between 1 and {output_limit}.")
        if not app.state.admission.try_acquire():
            return _error(429, "service_busy", "The service is at its concurrency limit.")
        work = asyncio.create_task(run_ask(payload, top_k, answerer))
        pending_asks.add(work)
        work.add_done_callback(pending_asks.discard)
        return await asyncio.shield(work)

    async def run_investigation(
        payload: InvestigateRequest,
        active_agent: DocumentAgent,
        stop: threading.Event,
        observe: Callable[[AgentProgress], None] | None = None,
    ) -> JSONResponse:
        try:
            outcome = await run_in_threadpool(
                active_agent.run, payload.query, cancelled=stop.is_set, observe=observe
            )
            content = {
                "status": outcome.status,
                "message": AGENT_MESSAGES[outcome.status],
                "blocks": [asdict(block) for block in outcome.blocks],
                "sources": [
                    AnswerSourceResponse(
                        citation_id=row.citation_id,
                        rank=row.ranked.rank,
                        title=row.title,
                        text=row.ranked.passage.text,
                        source_url=_safe_source_url(row.ranked.passage.metadata.get("source_url")),
                    ).model_dump()
                    for row in outcome.evidence
                ],
                "clarification": outcome.clarification,
                "events": [asdict(event) for event in outcome.events],
                "usage": {
                    "model_calls": outcome.model_calls,
                    "search_calls": outcome.search_calls,
                    "read_calls": outcome.read_calls,
                    "prompt_estimated_tokens": outcome.prompt_estimated_tokens,
                },
                "total_seconds": outcome.total_seconds,
                "agent_profile": outcome.profile_fingerprint,
            }
            failed = outcome.status in {
                "invalid_action",
                "invalid_answer",
                "generation_failed",
                "generation_timeout",
                "retrieval_failed",
            }
            return JSONResponse(
                content=content,
                status_code=503 if failed else 200,
                headers=_no_store_headers(),
            )
        except (Exception, SystemExit):
            return _error(503, "agent_failed", "Document investigation failed.")
        finally:
            app.state.admission.release()

    @app.post("/api/investigate", response_model=None)
    async def investigate(payload: InvestigateRequest, request: Request) -> JSONResponse:
        if agent is None:
            return _error(503, "agent_unavailable", "Document investigation is not enabled.")
        if not app.state.admission.try_acquire():
            return _error(429, "service_busy", "The service is at its concurrency limit.")
        stop = threading.Event()
        work = asyncio.create_task(run_investigation(payload, agent, stop))
        pending_investigations.add(work)
        work.add_done_callback(pending_investigations.discard)
        try:
            while not work.done():
                await asyncio.wait({work}, timeout=0.1)
                if await request.is_disconnected():
                    stop.set()
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            stop.set()
            raise

    @app.post("/api/investigate/stream", response_model=None)
    async def investigate_stream(payload: InvestigateRequest) -> StreamingResponse | JSONResponse:
        if agent is None:
            return _error(503, "agent_unavailable", "Document investigation is not enabled.")
        if not app.state.admission.try_acquire():
            return _error(429, "service_busy", "The service is at its concurrency limit.")
        feed = ProgressFeed()
        work = asyncio.create_task(run_investigation(payload, agent, feed.stop, feed.observe))
        pending_investigations.add(work)
        work.add_done_callback(pending_investigations.discard)
        return feed.response(work)

    # Optional compiled React client. The original root page and API contracts stay available.
    workbench_dir = frontend_dir or Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if (workbench_dir / "index.html").is_file():
        app.mount("/workbench", StaticFiles(directory=workbench_dir, html=True), name="workbench")

    return app
