"""Read-plan (chunk selection) evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.read_plan import (
    ReadPlanCase,
    evaluate_read_plan,
    load_read_plan_cases,
)


def _artifact(tmp_path: Path, doc_id: str, chunks: list[tuple[str, str, str]]) -> Path:
    """chunks: (chunk_id, title, text); hierarchy_path follows the title."""
    parsed = tmp_path / "parsed"
    parsed.mkdir(exist_ok=True)
    payload = {
        "document": {"doc_id": doc_id, "source_path": "x.md", "title": doc_id, "content": "", "metadata": {}},
        "chunks": [
            {
                "chunk_id": cid,
                "doc_id": doc_id,
                "chunk_index": index,
                "text": text,
                "source_path": "x.md",
                "title": title,
                "metadata": {"kb_id": "default", "hierarchy_path": [title]},
            }
            for index, (cid, title, text) in enumerate(chunks)
        ],
        "structured_document": None,
    }
    path = parsed / f"{doc_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return parsed


def test_chunk_recall_is_one_when_the_expected_fragment_is_read(tmp_path) -> None:
    parsed = _artifact(tmp_path, "d1", [("c1", "A", "text one"), ("c2", "B", "text two")])
    report = evaluate_read_plan(
        [ReadPlanCase(sample_id="s", doc_id="d1", expected_text="text two", focus_sections=("B",))],
        parsed_dir=parsed,
    )
    assert report["aggregate"]["chunk_recall"] == 1.0


def test_sibling_chunk_of_the_same_section_is_read(tmp_path) -> None:
    """The historical defect: one focus section, two chunks, only the first read.

    chunk_recall would pass (the fragment is found) but section_coverage exposes
    the dropped sibling - which is why both metrics exist.
    """
    parsed = _artifact(
        tmp_path,
        "d1",
        [
            ("c1", "Reimbursement", "file within 15 days"),
            ("c2", "Reimbursement", "attach receipts"),
            ("c3", "Leave", "sick leave needs a note"),
        ],
    )
    report = evaluate_read_plan(
        [ReadPlanCase("s", "d1", "attach receipts", focus_sections=("Reimbursement",))],
        parsed_dir=parsed,
    )
    assert report["aggregate"]["chunk_recall"] == 1.0
    assert report["aggregate"]["section_coverage_avg"] == 1.0
    assert report["results"][0]["selected_chunk_ids"] == ["c1", "c2"]


def test_missing_expected_fragment_is_reported_not_scored_as_a_miss(tmp_path) -> None:
    parsed = _artifact(tmp_path, "d1", [("c1", "A", "text one")])
    report = evaluate_read_plan(
        [ReadPlanCase("s", "d1", "nonexistent fragment", focus_sections=("A",))],
        parsed_dir=parsed,
    )
    assert report["aggregate"]["rows_missing_expected_chunk"] == 1
    assert report["aggregate"]["chunk_recall"] is None
    assert "not found in any chunk" in report["results"][0]["error"]


def test_unresolvable_focus_section_is_recorded_per_row(tmp_path) -> None:
    parsed = _artifact(tmp_path, "d1", [("c1", "A", "text one")])
    report = evaluate_read_plan(
        [ReadPlanCase("s", "d1", "text one", focus_sections=("No Such Section",))],
        parsed_dir=parsed,
    )
    assert report["aggregate"]["rows_with_error"] == 1
    assert "do not match any section" in report["results"][0]["error"]


def test_without_focus_sections_the_whole_document_is_read(tmp_path) -> None:
    parsed = _artifact(
        tmp_path, "d1", [("c1", "A", "one"), ("c2", "B", "two"), ("c3", "C", "three")]
    )
    report = evaluate_read_plan(
        [ReadPlanCase("s", "d1", "three")], parsed_dir=parsed
    )
    assert report["aggregate"]["chunk_recall"] == 1.0
    # No sections named -> no section-modelled denominator to report.
    assert report["aggregate"]["section_coverage_avg"] is None


def test_missing_artifact_raises(tmp_path) -> None:
    from app.core.exceptions import StoreError

    (tmp_path / "parsed").mkdir()
    with pytest.raises(StoreError, match="ingest the corpus first"):
        evaluate_read_plan(
            [ReadPlanCase("s", "absent", "x")], parsed_dir=tmp_path / "parsed"
        )


def test_dataset_loader_validates_rows(tmp_path) -> None:
    path = tmp_path / "rp.jsonl"
    path.write_text(
        json.dumps({"doc_id": "d1", "expected_text": "x", "focus_sections": "A"}) + "\n",
        encoding="utf-8",
    )
    cases = load_read_plan_cases(path)
    assert cases[0].focus_sections == ("A",)

    path.write_text(json.dumps({"doc_id": "d1"}) + "\n", encoding="utf-8")
    from app.core.exceptions import StoreError

    with pytest.raises(StoreError, match="expected_text"):
        load_read_plan_cases(path)
