# nano-rag

Enterprise RAG system with a real-data runtime: no frontend hardcoded business data, no runtime mock mode, and no silent backend fallback paths.

## Principles

- Real backend data is the source of truth for workspaces, ingest sources, documents, traces, eval reports, and diagnosis targets.
- Runtime failures should be visible. Missing provider keys, invalid model credentials, parser errors, and PostgreSQL/graph-store errors must not be hidden behind mock or fallback behavior.
- **No degradation paths.** There are no fallback, partial-result, or "continue without X" paths anywhere in the runtime: graph/wiki/store failures fail the request or the ingest job; LLM JSON that fails strict parsing raises `ModelOutputError`; corrupt parsed artifacts raise instead of being skipped; malformed config values raise at startup or first use instead of falling back to defaults. Optional auxiliary stages (page rendering, image extraction, graph extraction) either run and succeed or fail loudly when enabled.
- Semantic RAG behavior must not depend on hardcoded keyword lists for business meaning, discourse roles, conflict detection, routing, or answer policy. Use model-produced structure, parser metadata, typed configuration, or trace-visible behavior instead.
- Retrieval is document-level discovery, not dense-vector similarity: a document-scoped BM25 wiki is the retrieval engine's first hop, followed by deterministic version filtering and an LLM read plan. There is no dense-vector fallback, and no cross-encoder rerank stage: ranking quality comes from the lexical index and the model's reading plan, never from a second model.
- The frontend does not store or submit business API keys. Browser requests go through the account system or the Docker nginx proxy, which injects the local backend key — for business routes only; operator routes are authenticated by the caller with the admin key and are never auto-authenticated by the proxy.
- Every RAG answer should be traceable to retrieved context, citations, source documents, and trace IDs. This is checked, not just prompted: `app/generation/citation_check.py` deterministically verifies that every cited label resolves to a provided context and that the numbers a cited sentence states appear in the context it cites; the result lands on the trace (`citation_integrity`) and as a deterministic eval metric.
- Deterministic ingest behavior: **the multimodal document parser is the only parsing path** - there is no local extraction at all (no pypdf text mining, no Office/zip walking, no HTML stripping), so every accepted file is turned into Markdown by the provider's model. Text-like files are sent to the model as text (a text-only model can parse them); binary formats need a multimodal model and fail loudly otherwise. An unconfigured or disabled parser makes ingest fail; it never falls back to a lesser local parse. Provenance fields are honest (unknown page numbers stay `None`).
- Evaluation has two real modes: deterministic builtin metrics (token-based exact match + reference context recall, no LLM) and LLM-assisted deepeval metrics. Eval result rows pass every input field through; report gates fail when a gated metric is missing from the report.

## Tech Stack

- **Backend**: Python 3.12 + FastAPI + Uvicorn
- **Frontend**: React 19 + TypeScript + Vite, served by nginx in Docker
- **Discovery index**: document-level BM25 over the compiled wiki (no vector DB, no embedding model)
- **Graph store**: native PostgreSQL (document structure for optional graph expansion)
- **Model Providers**: ONE provider, two capabilities of the same family: a generation model does all the reasoning (query understanding, reading plan, evidence audit, graph extraction, answers) and a multimodal model parses every document. No embedding model, no rerank model, no local text extraction, no Bifrost/LiteLLM runtime layer.
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
  ingestion/    # Model-only parsing (no local extraction), chunking, metadata extraction
  model_client/ # Generation + document parser clients (single model; no rerank/embedding client)
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

Document -> multimodal model parser (only parsing path) -> normalizer -> chunker -> metadata -> committed parsed artifact + PostgreSQL graph rows + wiki discovery page

Ingest is incremental by content, not by timestamp. A file whose committed artifact already records the same SHA-256 and the same ingest fingerprint (parser provider/model + chunk configuration + artifact schema) is skipped without parsing or re-indexing, and reported in `unchanged_doc_ids` (job field `unchanged_documents`) so a skip is never confusable with "found nothing". `force=true` re-ingests deliberately. Parsed text is also kept in a content-addressed cache (`data/parsed/.parse_cache/<fingerprint>/<hash>.json`), so identical bytes at a *new* path are parsed once too; entries are namespaced by the fingerprint, which makes a parser or chunking change unable to read stale output.

Document identity is content-addressed on ingest: a file that arrives under a new path (upload, rename, reorganized corpus) but whose text equals an existing document in the same KB supersedes it instead of adding a second copy. Duplicates are not cosmetic — each copy takes a discovery slot and splits BM25 scores, which measurably pushes the real answer out of the top-k. Superseded ids are reported on the ingest response; `scripts/dedup_corpus.py` cleans a corpus that already accumulated them.

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
- `DOCUMENT_PARSER_API_KEY` - document parser key (required: the only parsing path)
- `DOCUMENT_PARSER_PROVIDER` - `gemini` or `qwen`; `qwen` covers DashScope and OpenAI-compatible vLLM
- `RAG_GRAPH_BACKEND` - `postgres` (standard Docker runtime) or `artifact` (bare local scripts, no database); also `none`/`disabled`
- `PG_URI` - PostgreSQL connection string for the graph store (defaults to the in-network `postgres` service)
- `RAG_WIKI_ENABLED` - the BM25 discovery layer; must be `true` (it is the retrieval engine, with no dense fallback)
- `RAG_PG_POOL_MIN` / `RAG_PG_POOL_MAX` - PostgreSQL pool bounds per process (app default 8; workers default 2 via `RAG_WORKER_PG_POOL_MAX` so 32 workers stay under `max_connections=100`)
- `RAG_BM25_TOKENIZER` - CJK segmentation for the discovery index: `cjk_ngram` (default: characters + bigrams), `cjk_bigram` (bigrams only; best measured separation, but a one-character query cannot match), or `jieba` (dictionary words)
- `RAG_BM25_MAX_DOCUMENTS` - discovery index capacity per process; exceeding it fails ingest loudly (default 50000)
- `RAG_WIKI_MAX_PAGES_PER_KB` - per-KB source page quota enforced at ingest (default 20000)
- `GENERATION_MAX_TOKENS` - generation output cap (0 = unset; invalid values fail startup)
- `RAG_INGEST_JOB_TTL_DAYS` - finished ingest-job record retention (default 7; 0 = keep forever)
- `RAG_INGEST_CONCURRENCY` - documents prepared at once during ingest (default 4; each is two model round trips). `1` restores serial behaviour
- `RAG_PARSE_CACHE_ENABLED` - content-addressed parse cache (default `true`); off means every file is re-parsed by the model
- `RAG_API_KEYS` - backend business API keys (single key: nginx injects exactly one; `RAG_ADMIN_API_KEYS` must differ)
- `RAG_PROXY_API_KEY` - derived from `RAG_API_KEYS` by compose; override only if the browser-facing key must differ

Configuration surface is deliberately two-tier: exactly two *external* credentials (generation, and the multimodal document parser, which is the only parsing path) plus `scripts/init_env.py`, which generates every internal secret (business/admin keys, postgres password) with `secrets.token_hex`. Derived values are never configured twice: `PG_URI` comes from `POSTGRES_*`, `RAG_PROXY_API_KEY` from `RAG_API_KEYS`. Legacy names (`COMPOSE_*`, `MODEL_GATEWAY_*`, `PG_PASSWORD`) are still recognised and migrated in place. Internal components (redis, postgres) are `expose`-only, so they need no operator-managed credentials.

## Code Conventions

- Pydantic v2 for schemas (`model_validate`, not `parse_obj`)
- Async FastAPI/httpx paths; blocking CPU/subprocess work goes through `asyncio.to_thread`
- `AppContainer` owns dependency wiring in lifespan
- YAML config supports env substitution
- OpenTelemetry spans should wrap pipeline stages
- Do not introduce frontend hardcoded business data or backend fallback behavior
- Do not introduce hardcoded semantic keyword lists in the RAG layer. If a degraded path seems necessary, the answer is to make the failure loud, not to degrade silently.

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 (internal) | FastAPI backend |
| frontend | 3001 | nginx + React SPA (public entry) |
| postgres | 5432 (internal) | Graph store |
| redis | 6379 (internal) | Celery broker + result backend |
| worker | - | Celery ingest worker |
