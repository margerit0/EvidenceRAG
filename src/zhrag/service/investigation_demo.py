"""Deterministic original synthetic backend for the complete restart demo.

No environment file, credentials, provider, network client or real corpus.
The model chooses from the persisted prompt, so process restarts need no hidden
script cursor. This is an interaction demonstration, not a quality evaluation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI

from zhrag.agent import AgentSettings, DocumentAgent
from zhrag.investigation_store import fingerprint
from zhrag.retrieval.online import (
    OnlineRetrievalResult,
    OnlineSettings,
    RankedPassage,
    StageTimings,
)
from zhrag.service.app import create_app
from zhrag.store.base import Passage

DOCUMENTS = (
    (
        "版本准备",
        "【原创合成文档】星桥是虚构系统。升级前须确认当前版本与目标版本。甲版到乙版须先在隔离环境验证兼容性，不能直接操作生产环境。",
    ),
    (
        "回退演练",
        "【原创合成文档】星桥从甲版升级到乙版时，应保存配置快照并演练回退。兼容性检查或回退演练失败时，应暂停升级并补齐验证。本示例不承诺无停机。",
    ),
)
SNAPSHOT = fingerprint({"contract": "synthetic-starbridge-v1", "documents": DOCUMENTS})


class DemoRetriever:
    settings = OnlineSettings(
        "synthetic-starbridge-v1", "none", "none", 2, 2, 2, 10, 2, 2, 2, rerank_enabled=False
    )

    def retrieve(self, query: str) -> OnlineRetrievalResult:
        index = 1 if "回退" in query else 0
        title, text = DOCUMENTS[index]
        passage = Passage(
            f"synthetic-{index}",
            text,
            f"synthetic-{index}",
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            index,
            {"heading_path": title},
        )
        return OnlineRetrievalResult(
            query,
            self.settings.profile_name,
            "none",
            "none",
            (),
            (),
            (),
            (),
            (RankedPassage(1, 1, 1.0, passage),),
            StageTimings(0, 0, 0, 0, 0, 0, 0, 0),
        )


class DemoGenerator:
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.profile_fingerprint = fingerprint({"contract": "starbridge-model-v1", "delay": delay})

    def generate(self, _system: str, user: str) -> str:
        time.sleep(self.delay)
        prompt = json.loads(user)
        action: dict[str, object]
        evidence, candidates = prompt["evidence"], prompt["candidates"]
        if not candidates:
            action = {"action": "search_docs", "query": "星桥 版本准备"}
        elif not evidence:
            action = {"action": "read_passage", "evidence_id": 1}
        elif "用户补充：" not in prompt["question"]:
            action = {
                "action": "clarify",
                "question": "请补充星桥集群的当前版本与目标版本（演示可填：甲版升级到乙版）。",
            }
        elif len(candidates) < 2:
            action = {"action": "search_docs", "query": "星桥 回退演练"}
        elif len(evidence) < 2:
            action = {"action": "read_passage", "evidence_id": 2}
        else:
            action = {
                "action": "answer",
                "answer": {
                    "answerable": True,
                    "blocks": [
                        {
                            "text": "星桥从甲版升级到乙版前，应先在隔离环境验证兼容性。",
                            "citations": [1],
                        },
                        {
                            "text": "保存配置快照并演练回退；"
                            "兼容性检查或回退演练失败时暂停升级并补齐验证。示例文档不承诺无停机。",
                            "citations": [2],
                        },
                    ],
                },
            }
        return json.dumps(action, ensure_ascii=False)


def create_demo_app(path: Path, *, delay: float = 2.0) -> FastAPI:
    retriever = DemoRetriever()
    agent = DocumentAgent(retriever, DemoGenerator(delay), SNAPSHOT, AgentSettings(max_seconds=180))
    return create_app(
        cast(Any, retriever),
        agent=agent,
        agent_models={"synthetic-demo": agent},
        published_index_identity=SNAPSHOT,
        investigation_db=path,
    )
