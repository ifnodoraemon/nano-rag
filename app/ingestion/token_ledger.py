"""Token accounting for one ingest batch.

Ingest spends model tokens in two places - parsing every document, and extracting
its knowledge graph - and neither was visible anywhere: an operator could see how
slow an ingest was, never what it cost.

The ledger is a ``ContextVar`` rather than an attribute on the clients on purpose:
document preparation runs concurrently (``RAG_INGEST_CONCURRENCY``), and a shared
counter would interleave the batches of different documents and attribute a
parser call to whichever document happened to be finishing. A context variable is
per-task, so each document accrues its own total with no locking and no
cross-talk.

Usage is best-effort by nature (a provider may omit it); a missing count is
recorded as zero and the caller can see that nothing was reported rather than
being told a fabricated number.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass
class TokenLedger:
    """Accumulates token usage for one unit of work (one document)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Tracked explicitly: some providers report only total_tokens, and deriving it
    # from prompt+completion would silently report zero cost for those.
    total: int = 0
    calls: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total

    def add(self, stage: str, usage: Mapping[str, Any] | None) -> None:
        if not usage:
            self.calls += 1
            self.by_stage[stage] = self.by_stage.get(stage, 0)
            return
        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        total = _as_int(usage.get("total_tokens"))
        if not total:
            total = prompt + completion
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total += total
        self.calls += 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + total

    def merge(self, other: "TokenLedger") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total += other.total
        self.calls += other.calls
        for stage, total in other.by_stage.items():
            self.by_stage[stage] = self.by_stage.get(stage, 0) + total

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
            "calls": self.calls,
            "by_stage": dict(self.by_stage),
        }


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


_current_ledger: ContextVar[TokenLedger | None] = ContextVar(
    "ingest_token_ledger", default=None
)


@contextmanager
def collect_tokens() -> Iterator[TokenLedger]:
    """Collect token usage for the enclosed work in the current task context."""
    ledger = TokenLedger()
    token = _current_ledger.set(ledger)
    try:
        yield ledger
    finally:
        _current_ledger.reset(token)


def record_tokens(stage: str, usage: Mapping[str, Any] | None) -> None:
    """Record usage when a ledger is being collected; a no-op otherwise."""
    ledger = _current_ledger.get()
    if ledger is not None:
        ledger.add(stage, usage)
