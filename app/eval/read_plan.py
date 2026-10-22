"""Evaluation of the deterministic half of the read plan: chunk selection.

The agentic read plan has two halves. The model picks documents and names focus
sections; then ``select_chunks_for_focus`` deterministically decides which chunks
of those documents get read. Only the second half can be evaluated without model
calls, and it is exactly where a real defect lived: a focus term was consumed
after its first matching chunk, so a heading spanning several chunks contributed
one chunk and the rest were silently dropped. The model then answered "not found"
for evidence that existed in a sibling chunk.

No metric in the suite could see that. Retrieval recall counted the document as
found - correctly - and answer metrics showed a wrong answer with no explanation.
``chunk_recall`` closes that gap: given a document and an expected fragment of its
text, was the chunk holding that fragment actually selected for reading?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.agentic.discovery import select_chunks_for_focus
from app.core.exceptions import StoreError


@dataclass
class ReadPlanCase:
    """One document-level question about chunk selection."""

    sample_id: str
    doc_id: str
    # Verbatim fragment of the passage that must be read to answer the question.
    expected_text: str
    focus_sections: tuple[str, ...] = ()
    kb_id: str = "default"


@dataclass
class ReadPlanRow:
    case: ReadPlanCase
    selected_chunk_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    # Every chunk of the sections the plan named: the section-sized denominator
    # that exposes "we read one chunk of three".
    section_chunk_ids: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def chunk_recall(self) -> float | None:
        """Did we read any chunk containing the expected text?"""
        if not self.expected_chunk_ids:
            return None
        return 1.0 if set(self.expected_chunk_ids) & set(self.selected_chunk_ids) else 0.0

    @property
    def section_coverage(self) -> float | None:
        """Fraction of the named sections' chunks that were read.

        This is the metric that would have caught the sibling-chunk defect: the
        expected fragment was found, one chunk was read, but coverage sat at 1/3.
        """
        if not self.section_chunk_ids:
            return None
        selected = set(self.selected_chunk_ids)
        covered = sum(1 for cid in self.section_chunk_ids if cid in selected)
        return covered / len(self.section_chunk_ids)


def load_read_plan_cases(path: str | Path) -> list[ReadPlanCase]:
    """Load a JSONL dataset of chunk-selection cases.

    Row shape:
        {"sample_id": "rp-1", "doc_id": "doc-...", "expected_text": "15 个自然日",
         "focus_sections": ["差旅报销"]}
    """
    cases: list[ReadPlanCase] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StoreError(f"read-plan dataset line {index + 1} is not JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise StoreError(f"read-plan dataset line {index + 1} is not an object")
        doc_id = str(raw.get("doc_id") or "").strip()
        expected_text = str(raw.get("expected_text") or "").strip()
        if not doc_id or not expected_text:
            raise StoreError(
                f"read-plan dataset line {index + 1} needs both doc_id and expected_text"
            )
        focus = raw.get("focus_sections") or []
        if isinstance(focus, str):
            focus = [focus]
        cases.append(
            ReadPlanCase(
                sample_id=str(raw.get("sample_id") or f"rp-{index + 1}"),
                doc_id=doc_id,
                expected_text=expected_text,
                focus_sections=tuple(str(term).strip() for term in focus if str(term).strip()),
                kb_id=str(raw.get("kb_id") or "default"),
            )
        )
    if not cases:
        raise StoreError(f"read-plan dataset {path} has no rows")
    return cases


def _chunk_id(chunk: dict, index: int) -> str:
    return str(chunk.get("chunk_id") or f"#{index}")


def _chunk_text(chunk: dict) -> str:
    return str(chunk.get("text") or "")


def _load_artifact(parsed_dir: Path, doc_id: str) -> dict:
    path = parsed_dir / f"{doc_id}.json"
    if not path.exists():
        raise StoreError(
            f"read-plan evaluation needs the parsed artifact for {doc_id} at {path}; "
            "ingest the corpus first"
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"parsed artifact {path} is unreadable: {exc}") from exc
    if not isinstance(artifact, dict) or not isinstance(artifact.get("chunks"), list):
        raise StoreError(f"parsed artifact {path} has no chunk list")
    return artifact


def evaluate_read_plan(
    cases: list[ReadPlanCase],
    *,
    parsed_dir: Path,
    max_chunks_per_doc: int = 24,
) -> dict[str, object]:
    """Score chunk selection for each case. Deterministic: no model calls."""
    rows: list[ReadPlanRow] = []
    for case in cases:
        artifact = _load_artifact(parsed_dir, case.doc_id)
        chunks = [chunk for chunk in artifact["chunks"] if isinstance(chunk, dict)]
        row = ReadPlanRow(case=case)
        row.expected_chunk_ids = [
            _chunk_id(chunk, index)
            for index, chunk in enumerate(chunks)
            if case.expected_text in _chunk_text(chunk)
        ]
        if not row.expected_chunk_ids:
            row.error = (
                f"expected_text not found in any chunk of {case.doc_id}: "
                f"{case.expected_text[:60]!r}"
            )
            rows.append(row)
            continue
        try:
            selected = select_chunks_for_focus(
                chunks,
                set(case.focus_sections),
                doc_id=case.doc_id,
                max_chunks=max_chunks_per_doc,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per row, not fatal
            row.error = str(exc)
            rows.append(row)
            continue
        row.selected_chunk_ids = [
            _chunk_id(chunk, index) for index, chunk in enumerate(selected)
        ]
        if case.focus_sections:
            # Denominator: every chunk of the sections the plan named, whether or
            # not they were selected. Without focus sections the plan reads the
            # whole document, so coverage is trivially the whole document.
            row.section_chunk_ids = _section_chunks(chunks, case.focus_sections)
        rows.append(row)

    scored = [row for row in rows if row.error is None]
    chunk_recalls = [row.chunk_recall for row in scored if row.chunk_recall is not None]
    coverages = [row.section_coverage for row in scored if row.section_coverage is not None]

    aggregate: dict[str, object] = {
        "chunk_recall": round(sum(chunk_recalls) / len(chunk_recalls), 4) if chunk_recalls else None,
        "section_coverage_avg": (
            round(sum(coverages) / len(coverages), 4) if coverages else None
        ),
        "rows": len(rows),
        "rows_scored": len(scored),
        "rows_with_error": len(rows) - len(scored),
        "rows_missing_expected_chunk": sum(
            1 for row in rows if not row.expected_chunk_ids
        ),
    }
    return {
        "status": "ok",
        "mode": "deterministic",
        "stage": "read_plan",
        "records": len(rows),
        "aggregate": aggregate,
        "results": [
            {
                "sample_id": row.case.sample_id,
                "doc_id": row.case.doc_id,
                "focus_sections": list(row.case.focus_sections),
                "chunk_recall": row.chunk_recall,
                "section_coverage": row.section_coverage,
                "selected_chunk_ids": row.selected_chunk_ids,
                "expected_chunk_ids": row.expected_chunk_ids,
                "section_chunk_ids": row.section_chunk_ids,
                "error": row.error,
            }
            for row in rows
        ],
    }


def _section_chunks(chunks: list[dict], focus: tuple[str, ...]) -> list[str]:
    """Every chunk belonging to any section named by ``focus``."""
    from app.agentic.discovery import _section_matches  # noqa: PLC2701 - same module contract
    from app.agentic.discovery import as_dict, as_str_list

    matched: list[str] = []
    for index, chunk in enumerate(chunks):
        metadata = as_dict(chunk.get("metadata"))
        haystack = [*as_str_list(metadata.get("hierarchy_path")), str(chunk.get("title") or "")]
        if any(_section_matches(term, haystack) for term in focus):
            matched.append(_chunk_id(chunk, index))
    return matched
