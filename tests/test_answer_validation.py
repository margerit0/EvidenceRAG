"""Adversarial wire values use synthetic text; no stored provider responses."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from test_answering import ranked
from zhrag.answering import (
    AnswerSettings,
    AnswerValidationError,
    AnswerValidationReason,
    parse_answer,
    select_evidence,
)

CANARY = "PRIVATE_RESPONSE_CANARY"
VALID = {"answerable": True, "blocks": [{"text": "合成说明", "citations": [1]}]}


@pytest.mark.parametrize(
    "raw,reason",
    [
        (None, "reply_type"),
        ("x" * 24_001, "reply_size"),
        ("```json\n{}\n```", "json_syntax"),
        ('{"answerable":true,"blocks":', "json_syntax"),
        ('{"' + CANARY + '":1,"' + CANARY + '":2}', "json_duplicate_key"),
        ('{"answerable":NaN,"blocks":[]}', "json_nonfinite"),
        ('{"answerable":true,"blocks":[{"text":"\\ud800","citations":[1]}]}', "invalid_unicode"),
        (json.dumps(json.dumps(VALID)), "answer_object"),
        (json.dumps({"answerable": True}), "answer_missing_field"),
        (json.dumps({**VALID, CANARY: CANARY}), "answer_extra_field"),
        (json.dumps({**VALID, "answerable": "true"}), "answerable_type"),
        (json.dumps({**VALID, "blocks": {}}), "blocks_type"),
        (json.dumps({**VALID, "answerable": False}), "refusal_with_blocks"),
        (json.dumps({**VALID, "blocks": []}), "block_count"),
        (json.dumps({**VALID, "blocks": [CANARY]}), "block_object"),
        (json.dumps({**VALID, "blocks": [{"text": CANARY}]}), "block_missing_field"),
        (
            json.dumps({**VALID, "blocks": [{**VALID["blocks"][0], CANARY: CANARY}]}),
            "block_extra_field",
        ),
    ],
)
def test_malformed_wire_data_has_fixed_reason_and_no_private_detail(
    raw: object, reason: str
) -> None:
    context = select_evidence("q", [ranked("合成证据")], AnswerSettings())
    with pytest.raises(AnswerValidationError) as raised:
        parse_answer(raw, context, AnswerSettings())  # type: ignore[arg-type]
    assert raised.value.reason.value == reason
    assert raised.value.code == str(raised.value) == "invalid_answer"
    assert CANARY not in repr(raised.value) + repr(vars(raised.value))


@pytest.mark.parametrize(
    "text,citations,reason",
    [
        (" ", [1], "block_text"),
        (1, [1], "block_text"),
        (CANARY + " [1]", [1], "inline_citation"),
        (CANARY, "1", "citations_type"),
        (CANARY, [], "citations_empty"),
        (CANARY, ["1"], "citation_type"),
        (CANARY, [True], "citation_type"),
        (CANARY, [1.0], "citation_type"),
        (CANARY, [2], "citation_unread"),
        (CANARY, [1, 1], "citation_duplicate"),
    ],
)
def test_invalid_later_block_rejects_entire_answer_without_coercion(
    text: object, citations: object, reason: str
) -> None:
    raw = json.dumps(
        {"answerable": True, "blocks": [VALID["blocks"][0], {"text": text, "citations": citations}]}
    )
    context = select_evidence("q", [ranked("合成证据")], AnswerSettings())
    with pytest.raises(AnswerValidationError) as raised:
        parse_answer(raw, context, AnswerSettings())
    assert raised.value.reason.value == reason
    assert CANARY not in repr(vars(raised.value))


def test_read_evidence_and_size_limits_remain_mandatory() -> None:
    settings = AnswerSettings()
    context = select_evidence("q", [ranked("合成证据")], settings)
    cases = [
        (replace(context, evidence=()), settings, VALID, "no_read_evidence"),
        (
            context,
            replace(settings, max_blocks=1),
            {**VALID, "blocks": VALID["blocks"] * 2},
            "block_count",
        ),
        (context, replace(settings, max_answer_chars=1), VALID, "answer_size"),
    ]
    for evidence, limits, value, reason in cases:
        with pytest.raises(AnswerValidationError) as raised:
            parse_answer(json.dumps(value), evidence, limits)
        assert raised.value.reason.value == reason
    assert parse_answer(json.dumps(VALID), context, settings)[0].citations == (1,)
    assert (
        parse_answer('{"answerable":false,"blocks":[]}', replace(context, evidence=()), settings)
        == ()
    )


def test_arbitrary_exception_text_cannot_become_a_diagnostic_code() -> None:
    with pytest.raises(ValueError) as raised:
        AnswerValidationError(CANARY)  # type: ignore[arg-type]
    assert CANARY not in str(raised.value)
    assert all(reason.value.isidentifier() for reason in AnswerValidationReason)


@pytest.mark.parametrize(
    "error,reason", [(RecursionError(CANARY), "json_depth"), (ValueError(CANARY), "json_value")]
)
def test_decoder_resource_errors_keep_private_exception_text_out_of_diagnostics(
    monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str
) -> None:
    context = select_evidence("q", [ranked("合成证据")], AnswerSettings())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr("zhrag.answering.json.loads", fail)
    with pytest.raises(AnswerValidationError) as raised:
        parse_answer("{}", context, AnswerSettings())
    assert raised.value.reason.value == reason
    assert CANARY not in repr(raised.value) + repr(vars(raised.value))
