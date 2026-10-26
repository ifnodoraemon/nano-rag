"""Evaluation must survive a single row failing.

Observed during a 270-question run: one question hit GENERATION_MAX_TOKENS, asyncio.gather
without return_exceptions propagated it, and the batch produced NO report at all - 269 good
results discarded along with the one failure.
"""

from __future__ import annotations

import pytest

from app.eval import service as eval_service


async def _fake_materialize(container, index, record):  # noqa: ANN001, ANN003
    query = str(record.get("query"))
    if query == "boom":
        raise RuntimeError("generation truncated (finish_reason=length)")
    return {**record, "answer": "fine", "retrieved_contexts": []}


@pytest.mark.asyncio
async def test_one_failing_row_does_not_abort_the_batch(monkeypatch) -> None:
    monkeypatch.setattr(eval_service, "_materialize_one", _fake_materialize)
    records = [{"query": "ok1"}, {"query": "boom"}, {"query": "ok2"}]

    out = await eval_service.materialize_eval_records(
        object(), records, concurrency=3  # type: ignore[arg-type]
    )

    assert len(out) == 3, "every row must appear in the result"
    assert [row["query"] for row in out] == ["ok1", "boom", "ok2"], "input order preserved"

    failed = [row for row in out if row.get("error")]
    assert len(failed) == 1
    assert "generation truncated" in failed[0]["error"]
    assert failed[0]["sample_id"], "the failing row still identifies itself"


@pytest.mark.asyncio
async def test_a_failure_is_recorded_not_hidden(monkeypatch) -> None:
    """The row carries the error, so errored_records counts it instead of reporting clean."""
    monkeypatch.setattr(eval_service, "_materialize_one", _fake_materialize)

    out = await eval_service.materialize_eval_records(
        object(), [{"query": "boom"}], concurrency=1  # type: ignore[arg-type]
    )

    assert out[0]["error"].startswith("RuntimeError")
    assert "answer" not in out[0], "a failed row must not be given a fabricated answer"


@pytest.mark.asyncio
async def test_the_other_rows_still_run(monkeypatch) -> None:
    seen: list[str] = []

    async def tracking(container, index, record):  # noqa: ANN001, ANN003
        seen.append(str(record.get("query")))
        return await _fake_materialize(container, index, record)

    monkeypatch.setattr(eval_service, "_materialize_one", tracking)
    await eval_service.materialize_eval_records(
        object(),
        [{"query": "a"}, {"query": "boom"}, {"query": "c"}],
        concurrency=3,  # type: ignore[arg-type]
    )

    assert sorted(seen) == ["a", "boom", "c"]
