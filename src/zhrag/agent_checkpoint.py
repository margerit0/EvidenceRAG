"""Allowlisted JSON checkpoint codec. No pickle, credentials or hidden reasoning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from zhrag.agent import AgentEvent, PlanningEvent, _Run
from zhrag.agent_planning import (
    DecisionAssessment,
    EnvironmentDependency,
    EvidenceQuote,
    InvestigationPlan,
    Requirement,
    RequirementCoverage,
    UserFact,
)
from zhrag.answering import AnswerBlock, Evidence
from zhrag.retrieval.online import RankedPassage
from zhrag.store.base import Passage

_TYPES: dict[str, Any] = {
    cls.__name__: cls
    for cls in (
        _Run,
        AgentEvent,
        PlanningEvent,
        DecisionAssessment,
        EnvironmentDependency,
        EvidenceQuote,
        InvestigationPlan,
        Requirement,
        RequirementCoverage,
        UserFact,
        AnswerBlock,
        Evidence,
        RankedPassage,
        Passage,
    )
}


def pack(value: Any) -> Any:
    """Preserve tuples, integer evidence keys and immutable passage metadata."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            "fields": {
                field.name: pack(getattr(value, field.name))
                for field in fields(value)
                if field.name not in {"progress", "started"}
            },
        }
    if isinstance(value, Mapping):
        return {"type": "mapping", "items": [[pack(k), pack(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, set, list)):
        return {"type": type(value).__name__, "items": [pack(v) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("unsupported checkpoint value")


def unpack(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    name = value["type"]
    if name == "mapping":
        return {unpack(k): unpack(v) for k, v in value["items"]}
    if name in {"tuple", "set", "list"}:
        return {"tuple": tuple, "set": set, "list": list}[name](unpack(v) for v in value["items"])
    if name not in _TYPES:
        raise ValueError("unknown checkpoint type")
    args = {key: unpack(v) for key, v in value["fields"].items()}
    if name == "_Run":
        args["started"] = 0.0
    return _TYPES[name](**args)


def new_checkpoint(question: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "decision",
        "state": pack(_Run(question, 0)),
        "raw": None,
        "search_result": None,
        "error": None,
        "stage": "decide",
        "supplements_applied": 0,
        "recorder": {"seq": 0, "invocation": 0, "active": None},
    }
