from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

from test_compare_agent_script import load_script
from zhrag.eval.agent_task_drafts import assemble_tasks, normalize_for_match, verify_draft
from zhrag.io_utils import read_jsonl, write_json, write_jsonl, write_text

DOC = "# 标题\n\n- PITR 仅支持恢复到**全新的空集群**。详见[兼容性](br/compat.md)。\n"
SOURCE = "pingcap/docs-cn:br/overview.md"
OTHER = "pingcap/docs-cn:br/guide.md"
QUOTE = "PITR 仅支持恢复到全新的空集群。详见兼容性。"
NOT_INDEXED = {
    "reference_sources": [OTHER],
    "evidence_quotes": [{"source": OTHER, "quote": "另一篇"}],
}


def _draft(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": "br-01-simple",
        "category": "simple",
        "source_group": "backup-restore",
        "question": "合成问题：PITR 能恢复到已有数据的集群吗？",
        "expected_status": "answered",
        "acceptance_criteria": ["指出 PITR 仅支持恢复到全新的空集群"],
        "reference_sources": [SOURCE],
        "evidence_quotes": [{"source": SOURCE, "quote": QUOTE}],
        "author_notes": "synthetic",
    }
    row.update(overrides)
    return row


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, dict[str, str], frozenset[str]]:
    write_text(tmp_path / "documents/core_ops/br/overview.md", DOC)
    write_text(tmp_path / "documents/core_ops/br/guide.md", "# 指南\n\n另一篇。\n")
    paths = {SOURCE: "documents/core_ops/br/overview.md", OTHER: "documents/core_ops/br/guide.md"}
    return tmp_path, paths, frozenset({SOURCE})


def test_quote_matching_ignores_links_and_emphasis_but_not_content() -> None:
    assert normalize_for_match("**全新的**[空集群](x.md)") == normalize_for_match("全新的空集群")
    assert normalize_for_match("全新的空集群") != normalize_for_match("全新的集群")
    decorated = '### `x` <span class="version-mark">从 v8 引入</span>\n+ 默认值：[`1`](/a.md#b)'
    assert normalize_for_match(decorated) == normalize_for_match("x 从 v8 引入 默认值：[`1`]")
    shortcode = '- 例如：\n\n    {{< copyable "sql" >}}\n\n    ```sql\n    SET x=1;\n    ```'
    assert normalize_for_match(shortcode) == normalize_for_match("例如：\n```sql\nSET x=1;\n```")


def test_verified_draft_has_no_problems(
    corpus: tuple[Path, dict[str, str], frozenset[str]],
) -> None:
    root, paths, indexed = corpus
    problems = verify_draft(_draft(), corpus_root=root, local_paths=paths, indexed_sources=indexed)
    assert problems == []


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            {"evidence_quotes": [{"source": SOURCE, "quote": "PITR 支持恢复到任意集群"}]},
            "not found",
        ),
        (NOT_INDEXED, "not in published index"),
        ({"reference_sources": []}, "without reference_sources"),
        ({"category": "multi_document"}, "two distinct sources"),
        ({"evidence_quotes": []}, "has no quote"),
        ({"author_notes": "x", "extra": 1}, "schema mismatch"),
        ({"expected_status": "answered_maybe"}, "invalid category or expected_status"),
    ],
)
def test_mechanical_grounding_failures_are_reported(
    corpus: tuple[Path, dict[str, str], frozenset[str]], overrides: dict[str, object], message: str
) -> None:
    root, paths, indexed = corpus
    problems = verify_draft(
        _draft(**overrides), corpus_root=root, local_paths=paths, indexed_sources=indexed
    )
    assert any(message in problem.message for problem in problems), problems


def test_assembly_never_marks_reviewed_and_keeps_evidence_separate() -> None:
    tasks, evidence = assemble_tasks(
        [_draft()], split_plan={"backup-restore": "test"}, snapshot="synthetic@abc"
    )
    assert tasks[0]["reviewed"] is False and tasks[0]["reviewer"] == ""
    assert tasks[0]["split"] == "test" and tasks[0]["snapshot"] == "synthetic@abc"
    assert "evidence_quotes" not in tasks[0] and evidence[0]["task_id"] == "br-01-simple"
    with pytest.raises(ValueError, match="split assignment"):
        assemble_tasks([_draft()], split_plan={}, snapshot="s")


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path]:
    script = load_script("assemble_agent_tasks")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "TASKS_ROOT", tmp_path / "indexes/agent_eval")
    corpus = tmp_path / "corpus"
    write_text(corpus / "documents/core_ops/br/overview.md", DOC)
    write_jsonl(
        corpus / "corpus_manifest.jsonl",
        [{"path": "br/overview.md", "local_path": "documents/core_ops/br/overview.md"}],
    )
    write_json(tmp_path / "state.json", {"documents": {SOURCE: {}}})
    write_json(tmp_path / "drafts/br.json", [_draft()])
    return script, tmp_path


def _argv(root: Path, *extra: str) -> list[str]:
    return [
        "--drafts", str(root / "drafts"),
        "--output", str(root / "indexes/agent_eval/v2/tasks.jsonl"),
        "--corpus", str(root / "corpus"),
        "--index-state", str(root / "state.json"),
        "--snapshot", "synthetic@abc",
        "--split", "backup-restore=dev",
        *extra,
    ]  # fmt: skip


def test_cli_verifies_by_default_and_writes_only_once(
    cli: tuple[ModuleType, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    script, root = cli
    output = root / "indexes/agent_eval/v2/tasks.jsonl"
    assert script.main(_argv(root)) == 0 and not output.exists()
    assert script.main(_argv(root, "--write")) == 0
    rows = list(read_jsonl(output))
    assert rows[0]["reviewed"] is False and rows[0]["split"] == "dev"
    assert next(read_jsonl(output.with_name("draft_evidence.jsonl")))["evidence_quotes"]
    assert script.main(_argv(root, "--write")) == 1
    assert "refusing to overwrite" in capsys.readouterr().out


def test_cli_rejects_ungrounded_quotes_and_outside_paths(
    cli: tuple[ModuleType, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    script, root = cli
    bad = deepcopy(_draft())
    bad["evidence_quotes"] = [{"source": SOURCE, "quote": "文档里没有这句话"}]
    write_json(root / "drafts/br.json", [bad])
    assert script.main(_argv(root, "--write")) == 1
    out = capsys.readouterr().out
    assert "not found verbatim" in out and "nothing written" in out
    argv = _argv(root)
    argv[argv.index("--output") + 1] = str(root / "elsewhere.jsonl")
    assert script.main(argv) == 1
