from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    trace_id: str
    latency_seconds: float | None = None
    query: str | None = None
    kb_id: str | None = None
    tenant_id: str | None = None
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
    rewritten_query: str | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    hyde_query: str | None = None
    kb_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    sample_id: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    freshness_ranked_chunk_ids: list[str] = Field(default_factory=list)
    retrieved: list[dict] = Field(default_factory=list)
    reranked: list[dict] = Field(default_factory=list)
    freshness_ranked: list[dict] = Field(default_factory=list)
    contexts: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    supporting_claims: list[dict] = Field(default_factory=list)
    answer: str | None = None
    model_alias: str | None = None
    embedding_model_alias: str | None = None
    rerank_model_alias: str | None = None
    prompt_version: str | None = None
    prompt_messages: list[dict] = Field(default_factory=list)
    generation_finish_reason: str | None = None
    generation_usage: dict[str, object] = Field(default_factory=dict)
    retrieval_params: dict[str, object] = Field(default_factory=dict)
    step_latencies: dict[str, float] = Field(default_factory=dict)


class RetrievalDebugResponse(BaseModel):
    query: str
    retrieved: list[dict]
    reranked: list[dict]
    contexts: list[dict]
    trace_id: str | None = None
