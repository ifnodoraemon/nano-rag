from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    trace_id: str
    latency_seconds: float | None = None
    query: str | None = None
    kb_id: str | None = None
    session_id: str | None = None
    model_alias: str | None = None
    prompt_version: str | None = None
    context_count: int | None = None
    conflicting_context_count: int | None = None
    conflict_claim_count: int | None = None
    insufficiency_claim_count: int | None = None
    conditional_claim_count: int | None = None


class TraceRecord(BaseModel):
    trace_id: str
    latency_seconds: float | None = None
    query: str | None = None
    original_query: str | None = None
    rewritten_query: str | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    hyde_query: str | None = None
    kb_id: str | None = None
    session_id: str | None = None
    sample_id: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    freshness_ranked_chunk_ids: list[str] = Field(default_factory=list)
    retrieved: list[dict] = Field(default_factory=list)
    freshness_ranked: list[dict] = Field(default_factory=list)
    contexts: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    supporting_claims: list[dict] = Field(default_factory=list)
    answer: str | None = None
    model_alias: str | None = None
    prompt_version: str | None = None
    prompt_messages: list[dict] = Field(default_factory=list)
    generation_finish_reason: str | None = None
    generation_usage: dict[str, object] = Field(default_factory=dict)
    synthesis_usage: dict[str, object] = Field(default_factory=dict)
    model_calls: list[dict] = Field(default_factory=list)
    provider_attempts: list[dict] = Field(default_factory=list)
    grounded_claims: list[dict] = Field(default_factory=list)
    retrieval_params: dict[str, object] = Field(default_factory=dict)
    step_latencies: dict[str, float] = Field(default_factory=dict)
    # Deterministic citation audit (app/generation/citation_check.py): does every
    # cited label exist, and do the numbers a cited sentence states appear in the
    # context it cites. None on traces recorded before this check existed.
    citation_integrity: dict[str, object] | None = None
    # The synthesis model's own verdict on whether the contexts answer the
    # question. Structured on purpose: the abstention metrics read this instead of
    # pattern-matching refusal wording.
    answerable: bool | None = None


class RetrievalDebugResponse(BaseModel):
    query: str
    # Discovery output. This build has one model (generation) and therefore no
    # rerank stage, so no ``reranked`` copy is emitted - it would imply a step
    # that never ran.
    retrieved: list[dict]
    contexts: list[dict]
    trace_id: str | None = None
