"""Exercise the real chat adapter with synthetic GLM-shaped HTTP responses.

These scripted responses test dispatch and wire compatibility, not model judgment.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import replace
from typing import Any

import pytest

from test_agent import ANSWER, READ, SEARCH, InvestigationRetriever
from test_answering import ranked
from zhrag.agent import AgentSettings, DocumentAgent
from zhrag.providers.answering import ChatAnswerGenerator
from zhrag.providers.chat import ChatConfig
from zhrag.retrieval.online import OnlineRetrievalResult

PRIVATE = "PRIVATE_PROVIDER_REASONING_OR_ANSWER"


def wire_agent(
    *actions: object,
    settings: AgentSettings | None = None,
    retriever: InvestigationRetriever | None = None,
) -> tuple[DocumentAgent, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def transport(request: urllib.request.Request) -> bytes:
        payload = json.loads(request.data or b"{}")
        assert payload["model"] == "z-ai/glm-5.3"
        assert payload["reasoning_effort"] == "high"
        assert payload["response_format"] == {"type": "json_object"}
        assert "max_tokens" not in payload and "max_completion_tokens" not in payload
        sent.append(json.loads(payload["messages"][1]["content"]))
        return json.dumps(
            {
                "model": "z-ai/glm-5.3",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(actions[len(sent) - 1], ensure_ascii=False),
                            "reasoning_content": PRIVATE,
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")

    generator = ChatAnswerGenerator(
        ChatConfig("synthetic-key", "https://example.invalid", "z-ai/glm-5.3"),
        max_output_tokens=None,
        reasoning_effort="high",
        max_retries=9,
        transport=transport,
        sleep=lambda _: None,
    )
    return DocumentAgent(
        retriever or InvestigationRetriever(), generator, "synthetic", settings or AgentSettings()
    ), sent


@pytest.mark.parametrize(
    "answer,reason",
    [
        (json.dumps(ANSWER["answer"]), "answer_object"),
        ({"answerable": True}, "answer_missing_field"),
        ({"answerable": "true", "blocks": []}, "answerable_type"),
        ({"answerable": True, "blocks": [{"text": PRIVATE, "citations": ["1"]}]}, "citation_type"),
        ({"answerable": True, "blocks": [{"text": PRIVATE, "citations": [2]}]}, "citation_unread"),
        (
            {"answerable": True, "blocks": [{"text": PRIVATE + " [1]", "citations": [1]}]},
            "inline_citation",
        ),
    ],
)
@pytest.mark.parametrize("review", [False, True])
def test_wire_failure_retains_only_fixed_diagnostic_and_never_starts_review(
    answer: object, reason: str, review: bool
) -> None:
    agent, sent = wire_agent(
        SEARCH,
        READ,
        {"action": "answer", "answer": answer},
        settings=AgentSettings(review_answers=review),
    )
    result = agent.run("合成问题")
    assert result.status == "invalid_answer" and not result.blocks
    assert result.events[-1].validation_error == reason
    assert not any(e.action == "review_answer" for e in result.events)
    assert PRIVATE not in repr(result)
    assert len(sent) == 3


def test_wire_clarify_after_read_is_terminal_without_review_or_invented_branch() -> None:
    question = "你的合成设备使用自动模式还是人工模式？"
    agent, sent = wire_agent(
        SEARCH,
        READ,
        {"action": "clarify", "question": question},
        ANSWER,
        settings=AgentSettings(review_answers=True),
    )
    result = agent.run("合成设备启动慢，我应该调高启动配额吗？")
    assert result.status == "clarification_needed" and result.clarification == question
    assert not result.blocks and len(sent) == 3
    assert not any(e.action == "review_answer" or e.validation_error for e in result.events)


def test_wire_cross_document_answer_reads_mechanism_and_keeps_conditional_scope() -> None:
    passages = {
        "first": "合成设备有电池时可以自动启动；无电池时也可手动启动。",
        "mechanism": "合成设备通过飞轮蓄能启动。飞轮未锁定时检修可能发生回转。",
    }

    class SyntheticRetriever(InvestigationRetriever):
        def retrieve(self, query: str) -> OnlineRetrievalResult:
            return replace(
                super().retrieve(query), passages=(ranked(passages[query], doc_id=query),)
            )

    answer = {
        "action": "answer",
        "answer": {
            "answerable": True,
            "blocks": [
                {"text": "有电池时可以自动启动，无电池也可手动启动。", "citations": [1]},
                {"text": "飞轮蓄能提供启动动力；未锁定飞轮就检修有回转风险。", "citations": [2]},
            ],
        },
    }
    agent, sent = wire_agent(
        SEARCH,
        READ,
        {"action": "search_docs", "query": "mechanism"},
        {"action": "read_passage", "evidence_id": 2},
        answer,
        retriever=SyntheticRetriever(),
    )
    result = agent.run("合成设备如何启动、原理是什么、检修风险是什么？")
    assert result.status == "answered" and result.search_calls == result.read_calls == 2
    assert [b.citations for b in result.blocks] == [(1,), (2,)]
    assert [e["text"] for e in sent[-1]["evidence"]] == list(passages.values())
    assert not any(e.validation_error for e in result.events)
    assert PRIVATE not in repr(result)
