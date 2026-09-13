"""Local, review-gated task sets for comparing document investigation methods.

Question text and review material stay outside version control. Validation proves
schema/review metadata consistency, not correctness of a human annotation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

TASK_CONTRACT = "document-investigation-tasks-v1"
CATEGORIES = ("simple", "multi_document", "clarification", "unanswerable")
EXPECTED_STATUSES = ("answered", "clarification_needed", "insufficient_evidence")
METHODS = ("single_rag", "fixed_workflow", "document_agent")


def _text(value: object, name: str, *, limit: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid {name}")
    value.encode("utf-8")
    return value.strip()


@dataclass(frozen=True, slots=True)
class AgentTask:
    task_id: str
    category: str
    split: str
    source_group: str
    question: str
    expected_status: str
    acceptance_criteria: tuple[str, ...]
    reference_sources: tuple[str, ...]
    reviewed: bool
    reviewer: str
    snapshot: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> AgentTask:
        if set(row) != set(cls.__dataclass_fields__):
            raise ValueError("task schema mismatch")
        task_id = _text(row["task_id"], "task_id", limit=64)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", task_id):
            raise ValueError("task_id must be a canonical slug")
        category = _text(row["category"], "category")
        split = _text(row["split"], "split")
        expected = _text(row["expected_status"], "expected_status")
        if category not in CATEGORIES or split not in {"dev", "test"}:
            raise ValueError("invalid category or split")
        if expected not in EXPECTED_STATUSES:
            raise ValueError("invalid expected_status")
        reviewed = row["reviewed"]
        if type(reviewed) is not bool:
            raise ValueError("reviewed must be boolean")
        lists: dict[str, tuple[str, ...]] = {}
        for name in ("acceptance_criteria", "reference_sources"):
            value = row[name]
            if not isinstance(value, list) or len(value) > 20:
                raise ValueError(f"invalid {name}")
            lists[name] = tuple(_text(item, name) for item in value)
        reviewer, snapshot = row["reviewer"], row["snapshot"]
        if not isinstance(reviewer, str) or not isinstance(snapshot, str):
            raise ValueError("review metadata must be text")
        if len(reviewer) > 100 or len(snapshot) > 200:
            raise ValueError("review metadata too long")
        reviewer.encode("utf-8")
        snapshot.encode("utf-8")
        if reviewed and (
            not reviewer.strip()
            or not snapshot.strip()
            or not lists["acceptance_criteria"]
            or (expected == "answered" and not lists["reference_sources"])
        ):
            raise ValueError("reviewed tasks require reviewer, snapshot, rubric and answer sources")
        return cls(
            task_id,
            category,
            split,
            _text(row["source_group"], "source_group", limit=200),
            _text(row["question"], "question"),
            expected,
            lists["acceptance_criteria"],
            lists["reference_sources"],
            reviewed,
            reviewer.strip(),
            snapshot.strip(),
        )


def validate_tasks(
    rows: Iterable[Mapping[str, object]], *, require_reviewed: bool = False
) -> tuple[AgentTask, ...]:
    tasks: list[AgentTask] = []
    seen: set[str] = set()
    questions: set[str] = set()
    groups: dict[str, str] = {}
    for row in rows:
        task = AgentTask.from_mapping(row)
        normalized = " ".join(task.question.split()).casefold()
        if task.task_id in seen or normalized in questions:
            raise ValueError("duplicate task id or question")
        if task.source_group in groups and groups[task.source_group] != task.split:
            raise ValueError("source group leaks across development and test splits")
        if require_reviewed and not task.reviewed:
            raise ValueError("unreviewed task cannot enter a frozen evaluation")
        seen.add(task.task_id)
        questions.add(normalized)
        groups[task.source_group] = task.split
        tasks.append(task)
    if not tasks:
        raise ValueError("task set is empty")
    return tuple(tasks)


def task_summary(tasks: tuple[AgentTask, ...]) -> dict[str, object]:
    """Aggregate-only manifest; never publish questions, rubrics, sources or reviewer names."""
    canonical = json.dumps(
        [asdict(task) for task in sorted(tasks, key=lambda item: item.task_id)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "contract": TASK_CONTRACT,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "task_count": len(tasks),
        "reviewed_count": sum(task.reviewed for task in tasks),
        "categories": dict(Counter(task.category for task in tasks)),
        "splits": dict(Counter(task.split for task in tasks)),
        "source_group_count": len({task.source_group for task in tasks}),
        "evaluation_ready": bool(tasks) and all(task.reviewed for task in tasks),
        "methods": list(METHODS),
    }


def draft_tasks() -> list[dict[str, object]]:
    """Author scenario scaffolds independently of corpus; NEVER label them reviewed."""
    topics = ("备份与恢复", "慢查询排查", "事务与锁", "内存使用", "数据导入", "执行计划")
    patterns = (
        ("simple", "answered", "关于 TiDB 的{topic}，请找出入门文档并说明适用范围。"),
        (
            "multi_document",
            "answered",
            "关于 TiDB 的{topic}，请综合不同章节列出前提、操作与验证步骤。",
        ),
        (
            "clarification",
            "clarification_needed",
            "我的 TiDB 在{topic}方面出了问题。直接告诉我该修改哪个配置值。",
        ),
        (
            "unanswerable",
            "insufficient_evidence",
            "我们未公开的生产集群昨天在{topic}方面是谁做了什么操作？请仅凭公开文档确认。",
        ),
    )
    rows: list[dict[str, object]] = []
    for group, topic in enumerate(topics, start=1):
        for category, expected, template in patterns:
            rows.append(
                {
                    "task_id": f"draft-{group:02d}-{category}",
                    "category": category,
                    "split": "test" if group > 4 else "dev",
                    "source_group": f"draft-topic-{group:02d}",
                    "question": template.format(topic=topic),
                    "expected_status": expected,
                    "acceptance_criteria": [],
                    "reference_sources": [],
                    "reviewed": False,
                    "reviewer": "",
                    "snapshot": "",
                }
            )
    return rows
