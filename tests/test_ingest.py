"""Ingestion identity, scope, and delta tests over synthetic fixtures.

No TiDB or CRUD-RAG bytes appear here: the manifest rows and documents are
written by the tests themselves, so the licensing boundary holds and the suite
stays hermetic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from zhrag.ingest import (
    STATE_SCHEMA,
    DocumentPlan,
    IngestState,
    ManifestDocument,
    Scope,
    chunk_id,
    chunker_fingerprint,
    diff_documents,
    document_loader,
    document_sha256,
    git_blob_sha1,
    load_manifest,
    plan_document,
    plan_documents,
    read_state,
    reusable_chunk_ids,
    scope_fingerprint,
    source_key,
    write_state,
)
from zhrag.io_utils import write_json, write_text

DOC = "---\ntitle: 标题\n---\n\n# 顶层\n\n正文内容。\n"
RELEASE = "---\ntitle: 发版\n---\n\n# 发版说明\n\n变更内容。\n"


def manifest_row(
    path: str,
    body: str,
    *,
    collection: str = "core_ops",
    theme: str = "deploy_config_tiup",
    category: str = "root",
) -> dict[str, object]:
    data = body.encode("utf-8")
    return {
        "path": path,
        "sha": git_blob_sha1(data),
        "size": len(data),
        "collection": collection,
        "theme": theme,
        "category": category,
        "source_url": f"https://example.invalid/{path}",
    }


def build_corpus(root: Path, rows: list[dict[str, object]], bodies: dict[str, str]) -> Path:
    manifest_path = root / "selected_manifest.json"
    write_json(manifest_path, {"documents": rows})
    for row in rows:
        path = root / "documents" / str(row["collection"]) / str(row["path"])
        write_text(path, bodies[str(row["path"])])
    return manifest_path


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        manifest_row("a.md", DOC),
        manifest_row("guide/b.md", DOC),
        manifest_row(
            "releases/release-1.0.md",
            RELEASE,
            collection="temporal_releases",
            theme="version_history",
            category="releases",
        ),
    ]
    bodies = {"a.md": DOC, "guide/b.md": DOC, "releases/release-1.0.md": RELEASE}
    manifest_path = build_corpus(tmp_path, rows, bodies)
    return manifest_path, tmp_path / "documents"


class TestIdentity:
    def test_source_key_is_namespaced_and_routing_independent(self) -> None:
        assert source_key("releases/x.md") == "pingcap/docs-cn:releases/x.md"

    @pytest.mark.parametrize("path", ["", "/abs.md", "win\\path.md"])
    def test_rejects_non_relative_posix_paths(self, path: str) -> None:
        with pytest.raises(ValueError, match="relative POSIX path"):
            source_key(path)

    def test_git_blob_sha1_matches_the_git_object_format(self) -> None:
        data = "中文\n".encode()
        expected = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\x00" + data,
            usedforsecurity=False,
        ).hexdigest()
        assert git_blob_sha1(data) == expected

    def test_chunk_id_is_framed_so_fields_cannot_run_together(self) -> None:
        # Without length framing, ("k1", 11, "t") and ("k", 111, "t") would hash
        # the same bytes and two different chunks would share a primary key.
        assert chunk_id("k1", 11, "t") != chunk_id("k", 111, "t")

    def test_chunk_id_tracks_the_exact_indexed_text(self) -> None:
        base = chunk_id("k", 0, "文本")
        assert base == chunk_id("k", 0, "文本")
        assert base != chunk_id("k", 0, "文本 ")
        assert base != chunk_id("k", 1, "文本")


class TestManifest:
    def test_loads_sorted_validated_rows(self, corpus: tuple[Path, Path]) -> None:
        manifest_path, _ = corpus
        documents = load_manifest(manifest_path)
        assert [document.path for document in documents] == [
            "a.md",
            "guide/b.md",
            "releases/release-1.0.md",
        ]
        assert documents[0].key == "pingcap/docs-cn:a.md"

    def test_rejects_duplicate_paths(self, tmp_path: Path) -> None:
        row = manifest_row("a.md", DOC)
        write_json(tmp_path / "m.json", {"documents": [row, dict(row)]})
        with pytest.raises(ValueError, match="duplicate manifest path"):
            load_manifest(tmp_path / "m.json")

    def test_rejects_missing_fields_and_empty_selection(self, tmp_path: Path) -> None:
        write_json(tmp_path / "m.json", {"documents": [{"path": "a.md"}]})
        with pytest.raises(ValueError, match="missing"):
            load_manifest(tmp_path / "m.json")
        write_json(tmp_path / "empty.json", {"documents": []})
        with pytest.raises(ValueError, match="selects no documents"):
            load_manifest(tmp_path / "empty.json")


class TestScopeAndPlanning:
    def test_evergreen_scope_excludes_temporal_releases(self, corpus: tuple[Path, Path]) -> None:
        manifest_path, documents_root = corpus
        manifest = load_manifest(manifest_path)
        planned = plan_documents(
            manifest,
            scope=Scope.evergreen(),
            load=document_loader(documents_root),
        )
        assert [plan.path for plan in planned] == ["a.md", "guide/b.md"]

    def test_scope_fingerprint_separates_selections(self) -> None:
        assert scope_fingerprint(Scope.evergreen()) != scope_fingerprint(
            Scope(exclude_collections=frozenset())
        )
        assert scope_fingerprint(Scope.evergreen()) == scope_fingerprint(Scope())

    def test_plan_carries_metadata_provenance_and_chunk_ids(
        self, corpus: tuple[Path, Path]
    ) -> None:
        manifest_path, documents_root = corpus
        [document] = [d for d in load_manifest(manifest_path) if d.path == "a.md"]
        plan = plan_document(document, document_loader(documents_root)(document))
        assert plan.key == "pingcap/docs-cn:a.md"
        assert plan.document_sha256 == document_sha256(DOC.encode("utf-8"))
        assert plan.metadata["collection"] == "core_ops"
        assert plan.metadata["theme"] == "deploy_config_tiup"
        assert plan.chunks
        for chunk in plan.chunks:
            assert chunk.chunk_id == chunk_id(plan.key, chunk.ordinal, chunk.contextual_text)
            assert len(chunk.chunk_id) == 64

    def test_size_and_checksum_mismatches_fail_closed(self, corpus: tuple[Path, Path]) -> None:
        manifest_path, _ = corpus
        [document] = [d for d in load_manifest(manifest_path) if d.path == "a.md"]
        with pytest.raises(ValueError, match="bytes"):
            plan_document(document, b"short")
        tampered = (DOC + "尾部").encode("utf-8")
        forged = ManifestDocument(
            path=document.path,
            blob_sha1=document.blob_sha1,
            size=len(tampered),
            collection=document.collection,
            theme=document.theme,
            category=document.category,
            source_url=document.source_url,
        )
        with pytest.raises(ValueError, match="!= file sha"):
            plan_document(forged, tampered)

    def test_missing_curated_file_names_the_document(self, corpus: tuple[Path, Path]) -> None:
        manifest_path, documents_root = corpus
        [document] = [d for d in load_manifest(manifest_path) if d.path == "a.md"]
        (documents_root / document.relative_path()).unlink()
        with pytest.raises(FileNotFoundError, match=r"a\.md"):
            document_loader(documents_root)(document)


class TestDelta:
    def plans(self, corpus: tuple[Path, Path]) -> tuple[DocumentPlan, ...]:
        manifest_path, documents_root = corpus
        return plan_documents(
            load_manifest(manifest_path),
            scope=Scope.evergreen(),
            load=document_loader(documents_root),
        )

    def state_from(
        self,
        planned: tuple[DocumentPlan, ...],
        **overrides: object,
    ) -> IngestState:
        return IngestState(
            scope=scope_fingerprint(Scope.evergreen()),
            chunker_fingerprint=chunker_fingerprint(target_tokens=400, hard_max_tokens=600),
            embedding_profile=str(overrides.get("embedding_profile", "profile-v1")),
            sparse_fingerprint="sparse-v1",
            collection_name="chunks_v1",
            documents={
                plan.key: {
                    "document_sha256": plan.document_sha256,
                    "metadata_fingerprint": plan.metadata_fingerprint,
                    "chunk_ids": list(plan.chunk_ids),
                }
                for plan in planned
            },
        )

    def test_first_build_is_all_added(self, corpus: tuple[Path, Path]) -> None:
        manifest_path, _ = corpus
        planned = self.plans(corpus)
        delta = diff_documents(
            planned,
            None,
            scope=Scope.evergreen(),
            manifest=load_manifest(manifest_path),
        )
        assert delta.counts["added"] == 2
        assert delta.counts["deleted"] == 0

    def test_unchanged_content_and_metadata_only_updates_are_distinct(
        self, corpus: tuple[Path, Path]
    ) -> None:
        manifest_path, _ = corpus
        manifest = load_manifest(manifest_path)
        planned = self.plans(corpus)
        previous = self.state_from(planned)
        delta = diff_documents(planned, previous, scope=Scope.evergreen(), manifest=manifest)
        assert delta.counts["unchanged"] == 2

        drifted = IngestState(
            scope=previous.scope,
            chunker_fingerprint=previous.chunker_fingerprint,
            embedding_profile=previous.embedding_profile,
            sparse_fingerprint=previous.sparse_fingerprint,
            collection_name=previous.collection_name,
            documents={
                key: {**dict(record), "metadata_fingerprint": "stale"}
                for key, record in previous.documents.items()
            },
        )
        delta = diff_documents(planned, drifted, scope=Scope.evergreen(), manifest=manifest)
        assert delta.counts["metadata_only"] == 2
        assert delta.counts["updated"] == 0

    def test_edited_document_is_updated_and_removed_one_is_deleted(self, tmp_path: Path) -> None:
        rows = [manifest_row("a.md", DOC), manifest_row("b.md", DOC)]
        manifest_path = build_corpus(tmp_path, rows, {"a.md": DOC, "b.md": DOC})
        manifest = load_manifest(manifest_path)
        planned = plan_documents(
            manifest,
            scope=Scope.evergreen(),
            load=document_loader(tmp_path / "documents"),
        )
        previous = self.state_from(planned)

        edited = DOC + "\n新增一段。\n"
        rows = [manifest_row("a.md", edited)]
        manifest_path = build_corpus(tmp_path, rows, {"a.md": edited})
        manifest = load_manifest(manifest_path)
        planned = plan_documents(
            manifest,
            scope=Scope.evergreen(),
            load=document_loader(tmp_path / "documents"),
        )
        delta = diff_documents(planned, previous, scope=Scope.evergreen(), manifest=manifest)
        assert delta.updated == ("pingcap/docs-cn:a.md",)
        assert delta.deleted == ("pingcap/docs-cn:b.md",)

    def test_a_renamed_path_is_a_delete_plus_an_add(self, tmp_path: Path) -> None:
        manifest_path = build_corpus(tmp_path, [manifest_row("old.md", DOC)], {"old.md": DOC})
        previous = self.state_from(
            plan_documents(
                load_manifest(manifest_path),
                scope=Scope.evergreen(),
                load=document_loader(tmp_path / "documents"),
            )
        )
        manifest_path = build_corpus(tmp_path, [manifest_row("new.md", DOC)], {"new.md": DOC})
        manifest = load_manifest(manifest_path)
        planned = plan_documents(
            manifest,
            scope=Scope.evergreen(),
            load=document_loader(tmp_path / "documents"),
        )
        delta = diff_documents(planned, previous, scope=Scope.evergreen(), manifest=manifest)
        assert delta.added == ("pingcap/docs-cn:new.md",)
        assert delta.deleted == ("pingcap/docs-cn:old.md",)

    def test_excluded_scope_is_never_reported_as_a_deletion(
        self, corpus: tuple[Path, Path]
    ) -> None:
        manifest_path, documents_root = corpus
        manifest = load_manifest(manifest_path)
        everything = plan_documents(
            manifest,
            scope=Scope(exclude_collections=frozenset()),
            load=document_loader(documents_root),
        )
        previous = self.state_from(everything)
        planned = self.plans(corpus)
        delta = diff_documents(planned, previous, scope=Scope.evergreen(), manifest=manifest)
        assert delta.deleted == ()
        assert delta.out_of_scope == ("pingcap/docs-cn:releases/release-1.0.md",)

    def test_an_incomplete_plan_stops_the_build_instead_of_deleting_rows(
        self, corpus: tuple[Path, Path]
    ) -> None:
        manifest_path, _ = corpus
        manifest = load_manifest(manifest_path)
        planned = self.plans(corpus)
        with pytest.raises(ValueError, match="were not planned"):
            diff_documents(
                planned[:1],
                self.state_from(planned),
                scope=Scope.evergreen(),
                manifest=manifest,
            )


class TestReuseAndState:
    def test_dense_reuse_requires_the_same_embedding_profile(
        self, corpus: tuple[Path, Path]
    ) -> None:
        manifest_path, documents_root = corpus
        planned = plan_documents(
            load_manifest(manifest_path),
            scope=Scope.evergreen(),
            load=document_loader(documents_root),
        )
        previous = IngestState(
            scope=scope_fingerprint(Scope.evergreen()),
            chunker_fingerprint=chunker_fingerprint(target_tokens=400, hard_max_tokens=600),
            embedding_profile="profile-v1",
            sparse_fingerprint="sparse-v1",
            collection_name="chunks_v1",
            documents={
                plan.key: {
                    "document_sha256": plan.document_sha256,
                    "metadata_fingerprint": plan.metadata_fingerprint,
                    "chunk_ids": list(plan.chunk_ids),
                }
                for plan in planned
            },
        )
        every_id = {chunk.chunk_id for plan in planned for chunk in plan.chunks}
        assert reusable_chunk_ids(planned, previous, embedding_profile="profile-v1") == every_id
        assert reusable_chunk_ids(planned, previous, embedding_profile="profile-v2") == frozenset()
        assert reusable_chunk_ids(planned, None, embedding_profile="profile-v1") == frozenset()

    def test_state_round_trips_and_rejects_foreign_files(self, tmp_path: Path) -> None:
        state = IngestState(
            scope="scope",
            chunker_fingerprint="chunker",
            embedding_profile="profile-v1",
            sparse_fingerprint="sparse",
            collection_name="chunks_v1",
            documents={
                "pingcap/docs-cn:a.md": {
                    "document_sha256": "a" * 64,
                    "metadata_fingerprint": "b" * 64,
                    "chunk_ids": ["c" * 64],
                }
            },
        )
        path = tmp_path / "state.json"
        write_state(path, state)
        loaded = read_state(path)
        assert loaded is not None
        assert loaded.collection_name == "chunks_v1"
        assert loaded.chunk_ids() == frozenset({"c" * 64})
        assert read_state(tmp_path / "absent.json") is None

        write_json(tmp_path / "other.json", {"schema": "something-else", "documents": {}})
        with pytest.raises(ValueError, match=STATE_SCHEMA):
            read_state(tmp_path / "other.json")

    def test_state_rejects_entries_without_a_content_hash(self, tmp_path: Path) -> None:
        write_json(
            tmp_path / "state.json",
            {"schema": STATE_SCHEMA, "documents": {"k": {"chunk_ids": []}}},
        )
        with pytest.raises(ValueError, match="document_sha256"):
            read_state(tmp_path / "state.json")
