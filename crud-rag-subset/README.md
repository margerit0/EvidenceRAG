# CRUD-RAG personal-project subset

This folder contains a reproducible, small subset of the official CRUD-RAG
data for a local Chinese RAG project.

## Contents

- `corpus/corpus.jsonl`: 500 news documents. Each line has `id`, `title`,
  `text`, and source metadata.
- `eval/qa_1doc.jsonl`: the matching 500 single-document QA records.
- `raw/split_merged.json`: the downloaded official split source file.
- `subset_manifest.json`: selection rules and provenance.
- `build_subset.ps1`: reruns the deterministic extraction.

The selected task is `questanswer_1doc` because it is the simplest self-
contained RAG workflow: every evaluation question has one evidence document
included in the corpus. The source file also contains multi-document QA,
summarization, continuation, and hallucination-modification tasks, which are
left out of this first project version.

## Rebuild

Run from this folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_subset.ps1
```

The script filters records with non-empty fields and article text between 500
and 8000 characters, then orders them by a stable SHA-256 score of the record
ID and takes the first 500. The same input therefore produces the same subset.

## Attribution and use

Source: [IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG).
Please retain the upstream attribution and review the source-data terms before
redistributing the news text publicly. The repository code is Apache-2.0, but
the underlying news content may have separate rights.
