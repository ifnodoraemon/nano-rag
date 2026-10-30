"""Request-scoped accounting shared by concurrent tasks, with per-call provenance."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class ModelUsage:
    calls: list[dict] = field(default_factory=list)
    max_calls: int = 32
    reserved: int = 0
    provider_attempts: list[dict] = field(default_factory=list)

    def total(self) -> dict:
        totals = {key: sum(int(c.get("usage", {}).get(key, 0) or 0) for c in self.calls)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        totals["total_tokens"] = sum(
            int(c.get("usage", {}).get("total_tokens", 0) or 0)
            or int(c.get("usage", {}).get("prompt_tokens", 0) or 0)
            + int(c.get("usage", {}).get("completion_tokens", 0) or 0)
            for c in self.calls
        )
        return {**totals, "model_calls": len(self.calls),
                "provider_attempts": len(self.provider_attempts),
                "retry_count": sum(a["attempt"] > 1 for a in self.provider_attempts),
                "structured_retries": sum(bool(c.get("retry_reason")) for c in self.calls),
                "usage_complete": bool(self.calls) and all(c.get("usage") for c in self.calls)
                    and all(a.get("status", 0) < 400 and a.get("status", 0) > 0 for a in self.provider_attempts)}


_current: ContextVar[ModelUsage | None] = ContextVar("rag_model_usage", default=None)
_stage: ContextVar[str] = ContextVar("rag_model_stage", default="generation")


@contextmanager
def collect_model_usage(max_calls: int = 32):
    ledger = ModelUsage(max_calls=max_calls)
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)


@contextmanager
def model_stage(name: str):
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)


def record_model_call(model: str, usage: dict | None, started: float, **details) -> None:
    ledger = _current.get()
    if ledger is not None:
        ledger.calls.append({"stage": _stage.get(), "model": model,
                             "usage": dict(usage or {}),
                             "seconds": round(perf_counter() - started, 6), **details})


def reserve_model_call():
    ledger = _current.get()
    if ledger is not None:
        if ledger.reserved >= ledger.max_calls:
            from app.core.exceptions import ModelOutputError
            raise ModelOutputError("request model-call budget exhausted")
        ledger.reserved += 1


def begin_provider_attempt(path: str, attempt: int) -> dict:
    record = {"stage": _stage.get(), "path": path, "attempt": attempt}
    ledger = _current.get()
    if ledger is not None:
        record["started"] = perf_counter()
        ledger.provider_attempts.append(record)
    return record
