from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes_debug import (
    get_benchmark_report_detail,
    get_eval_report_detail,
    run_eval,
)
from app.schemas.eval import EvalRunRequest


def _request_with_container(container) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=container))
    )


@pytest.mark.asyncio
async def test_eval_report_detail_rejects_path_outside_eval_dir() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_eval_report_detail("../secret.json")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_benchmark_report_detail_rejects_path_outside_benchmark_dir() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_benchmark_report_detail("../not-benchmark.json")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_run_eval_rejects_dataset_outside_eval_dir() -> None:
    container = SimpleNamespace(
        ragas_runner=SimpleNamespace(
            run=lambda records: {
                "status": "ok",
                "records": len(records),
                "aggregate": {},
                "results": [],
            }
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_eval(
            EvalRunRequest(dataset_path="../outside.jsonl"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 400
