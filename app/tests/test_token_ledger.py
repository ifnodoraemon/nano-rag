import pytest
from app.ingestion.token_ledger import TokenLedger, collect_tokens, record_tokens

def test_ledger_sums_and_groups_by_stage():
    ledger = TokenLedger()
    ledger.add("parse", {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    ledger.add("graph_extraction", {"prompt_tokens": 50, "completion_tokens": 10})
    assert ledger.total_tokens == 180
    assert ledger.calls == 2
    assert ledger.by_stage == {"parse": 120, "graph_extraction": 60}

def test_missing_usage_counts_as_a_call_not_a_fabricated_number():
    ledger = TokenLedger()
    ledger.add("parse", None)
    ledger.add("parse", {})
    assert ledger.total_tokens == 0
    assert ledger.calls == 2

def test_total_falls_back_to_prompt_plus_completion():
    ledger = TokenLedger()
    ledger.add("parse", {"prompt_tokens": 7, "completion_tokens": 3})
    assert ledger.total_tokens == 10

def test_record_is_scoped_and_silent_without_a_ledger():
    record_tokens("parse", {"total_tokens": 99})   # no ledger: must be a no-op
    with collect_tokens() as ledger:
        record_tokens("parse", {"total_tokens": 5})
        assert ledger.total_tokens == 5
    record_tokens("parse", {"total_tokens": 99})   # outside: ignored again
    assert ledger.total_tokens == 5

def test_nested_collect_does_not_leak_to_the_outer_ledger():
    with collect_tokens() as outer:
        with collect_tokens() as inner:
            record_tokens("parse", {"total_tokens": 5})
        record_tokens("parse", {"total_tokens": 7})
        assert inner.total_tokens == 5
        assert outer.total_tokens == 7

@pytest.mark.asyncio
async def test_concurrent_tasks_get_isolated_ledgers():
    import asyncio
    async def worker(n):
        with collect_tokens() as ledger:
            await asyncio.sleep(0.01)
            record_tokens("parse", {"total_tokens": n})
            await asyncio.sleep(0.01)
        return ledger.total_tokens
    got = await asyncio.gather(*(worker(n) for n in range(1, 21)))
    assert got == list(range(1, 21))
