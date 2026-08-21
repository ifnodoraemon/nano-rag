# nano-rag

Enterprise RAG system with a real-data runtime: no frontend hardcoded business data, no runtime mock mode, and no silent backend fallback paths.

## Principles

- Real backend data is the source of truth for workspaces, ingest sources, documents, traces, eval reports, and diagnosis targets.
- Runtime failures should be visible. Missing provider keys, invalid model credentials, parser errors, and PostgreSQL/graph-store errors must not be hidden behind mock or fallback behavior.
- Semantic RAG behavior must not depend on hardcoded keyword lists for business meaning, discourse roles, conflict detection, routing, or answer policy. Use model-produced structure, parser metadata, typed configuration, or trace-visible degraded behavior instead.
- Retrieval is document-level discovery, not dense-vector similarity: a document-scoped BM25 wiki is the retrieval engine's first hop, followed by deterministic version filtering and an LLM read plan. There is no dense-vector fallback.
- The frontend does not store or submit business API keys. Browser requests go through the account system or the Docker nginx proxy, which injects the local backend key.
- Every RAG answer should be traceable to retrieved context, citations, source documents, and trace IDs.

## Tech Stack

- **Backend**: Python 3.12 + FastAPI + Uvicorn
- **Frontend**: React 19 + TypeScript + Vite, served by nginx in Docker
- **Discovery index**: document-level BM25 over the compiled wiki (no vector DB, no embedding model)
- **Graph store**: native PostgreSQL (document structure for optional graph expansion)
- **Model Providers**: direct Gemini or Qwen provider configuration (generation + document parser + optional rerank); no Bifrost/LiteLLM runtime layer, no embedding endpoint
- **Tracing**: OpenTelemetry HTTP export to Langfuse
- **Testing**: pytest + pytest-asyncio, TypeScript build/lint
- **Evaluation**: deterministic eval plus optional deepeval metrics

## Project Structure

```text
app/
  agentic/      # LangGraph agentic pipeline: discovery -> read plan -> deep read -> synthesis
  api/          # FastAPI routes (business + debug)
  core/         # Config, exceptions, logging, tracing
  generation/   # Prompt builder, answer formatter
  ingestion/    # Document parsing, chunking, metadata extraction
  model_client/ # Generation, rerank, and document parser clients
  retrieval/    # BM25 index, graph store/index/expander, context builder, versioning
  schemas/      # Pydantic models
  wiki/         # Document-level wiki compiler (pages) + BM25 searcher (discovery layer)
configs/        # settings.yaml, models.yaml, prompts.yaml
frontend/       # React SPA source
docker/         # docker-compose.yml + Dockerfiles + nginx config
scripts/        # run_eval.py, run_benchmark.py, live_smoke.py
```

## Runtime Architecture

### Ingestion Pipeline

Document -> configured parser -> normalizer -> chunker -> metadata -> committed parsed artifact + PostgreSQL graph rows + wiki discovery page

### Retrieval (Agentic Discovery) Pipeline

Query -> wiki BM25 discovery (document-level) -> deterministic version filter (latest wins per source_key) -> LLM read plan (structured json_schema) -> deep-read selected parsed artifacts -> context builder (+ optional PostgreSQL graph expansion)

There is no embedding call and no dense/hybrid retrieval path. Discovery and the read plan each run on the configured generation gateway and are visible in the trace.

### Generation Pipeline

Context + query -> prompt builder -> configured generation gateway -> answer formatter with citations

## Commands

```bash
# Backend tests
python -m pytest app/tests

# Frontend checks
npm --prefix frontend run lint
npm --prefix frontend run build

# Standard runtime
docker compose -f docker/docker-compose.yml up -d --build
```

Do not use non-Docker commands to start the app for runtime validation.

## Key Environment Variables

- `MODEL_GATEWAY_MODE` - `live`; `mock` is not supported by the real-data runtime
- `GENERATION_API_BASE_URL` - generation provider endpoint
- `GENERATION_API_KEY` - generation provider key
- `GENERATION_MODEL_ALIAS` - generation model alias; default examples are Gemini or Qwen
- `DOCUMENT_PARSER_API_BASE_URL` - document parser endpoint
- `DOCUMENT_PARSER_API_KEY` - document parser key
- `DOCUMENT_PARSER_PROVIDER` - `gemini` or `qwen`; `qwen` covers DashScope and OpenAI-compatible vLLM
- `RAG_GRAPH_BACKEND` - `postgres` (standard Docker runtime) or `artifact` (bare local scripts, no database); also `none`/`disabled`
- `PG_URI` - PostgreSQL connection string for the graph store (defaults to the in-network `postgres` service)
- `RAG_WIKI_ENABLED` - the BM25 discovery layer; must be `true` (it is the retrieval engine, with no dense fallback)
- `RAG_API_KEYS` - backend business API keys

## Code Conventions

- Pydantic v2 for schemas (`model_validate`, not `parse_obj`)
- Async FastAPI/httpx paths
- `AppContainer` owns dependency wiring in lifespan
- YAML config supports env substitution
- OpenTelemetry spans should wrap pipeline stages
- Do not introduce frontend hardcoded business data or backend fallback behavior
- Do not introduce hardcoded semantic keyword lists in the RAG layer. If a degraded path is necessary for availability, it must be metadata-only, explicitly marked in trace output, and covered by tests.

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 (internal) | FastAPI backend |
| frontend | 3001 | nginx + React SPA (public entry) |
| postgres | 5432 (internal) | Graph store |
| redis | 6379 (internal) | Celery broker + result backend |
| worker | - | Celery ingest worker |
