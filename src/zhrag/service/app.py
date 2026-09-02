"""FastAPI application factory with no provider or database composition policy.

The online retriever is deliberately synchronous: its embedding, Milvus, and
rerank clients are blocking. The HTTP boundary runs it in Starlette's threadpool
and admits only a bounded number of requests, so one slow provider cannot turn
Milvus Lite's single local database into an unbounded work queue.
"""

from __future__ import annotations

import inspect
import math
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Protocol

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from zhrag.retrieval.online import OnlineRetrievalResult, OnlineRetriever, StageTimings
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


def create_app(  # noqa: PLR0915
    retriever: OnlineRetriever,
    *,
    info: ServiceInfo | None = None,
    static_dir: Path | None = None,
    max_concurrency: int = 1,
    search_runner: SearchRunner | None = None,
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

    return app
