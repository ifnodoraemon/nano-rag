# nano-rag

Enterprise RAG (Retrieval-Augmented Generation) system with hybrid retrieval, semantic chunking, and offline evaluation.

## Tech Stack

- **Backend**: Python 3.12 + FastAPI 0.116 + Uvicorn
- **Frontend**: React 19 + TypeScript 5.9 + Vite 8 + Zustand
- **Vector DB**: Milvus 2.5 (etcd + MinIO)
- **Model Gateway**: OpenAI-compatible (Gemini API default)
- **Tracing**: OpenTelemetry + Phoenix UI
- **Testing**: pytest + pytest-asyncio
- **Evaluation**: RAGAS 0.2

## Project Structure

```
app/
  api/          # FastAPI routes (business + debug)
  core/         # Config, exceptions, logging, tracing
  generation/   # Prompt builder, answer formatter, LLM service
  ingestion/    # Document parsing, chunking, metadata extraction
  model_client/ # Embedding, generation, rerank, document parser clients
  retrieval/    # Hybrid retrieval (BM25 + vector), reranking, context builder
  schemas/      # Pydantic models
  vectorstore/  # Milvus + in-memory repository
  tests/        # 31 test modules
  main.py       # Entry point
configs/        # settings.yaml, models.yaml, prompts.yaml
frontend/       # React SPA
docker/         # docker-compose.yml + Dockerfiles
scripts/        # run_eval.py, run_benchmark.py
```

## Architecture

### Ingestion Pipeline
Document -> Parser (Docling/pypdf) -> Normalizer -> Semantic Chunker -> Metadata -> Embedding -> Milvus

### Retrieval Pipeline
Query -> Query Rewriter -> Hybrid Retrieval (BM25 + Vector) -> Reranking -> Context Builder -> Freshness Filter

### Generation Pipeline
Context + Query -> Prompt Builder -> LLM -> Answer Formatter (with citation tracking)

## Key Commands

```bash
# Development
uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev

# Testing
pytest app/tests

# Docker
cd docker && docker-compose up -d

# Evaluation
python scripts/run_eval.py
python scripts/run_benchmark.py
```

## Configuration

- `configs/settings.yaml` - Chunking (800 tokens, 120 overlap), retrieval (top_k=20), timeouts
- `configs/models.yaml` - Model endpoints and aliases per capability
- `configs/prompts.yaml` - System prompts for generation
- `.env` - Runtime environment (gateway URLs, API keys, feature flags)

## Key Environment Variables

- `MODEL_GATEWAY_MODE` - live / mock
- `MODEL_GATEWAY_BASE_URL` - OpenAI-compatible endpoint
- `VECTORSTORE_BACKEND` - milvus / memory
- `DISABLE_RERANK` - 1 to skip reranking

## Code Conventions

- Pydantic v2 for all schemas (use `model_validate`, not `parse_obj`)
- Async throughout (FastAPI async routes, httpx async client)
- AppContainer for dependency injection (initialized in lifespan)
- YAML config with env var substitution (`${VAR_NAME}`)
- Structured logging via Python logging module
- OpenTelemetry spans for tracing all pipeline stages

## API Endpoints

- `GET /health` - Health check (gateway + vectorstore + phoenix)
- `POST /v1/rag/ingest` - Document ingestion
- `POST /v1/rag/chat` - RAG chat (with streaming support)
- `POST /v1/rag/feedback` - User feedback collection
- `POST /v1/rag/benchmark` - Performance benchmarking
- Business APIs require API key authentication (X-API-Key header)

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 | FastAPI backend |
| frontend | 3000 | Nginx + React SPA |
| milvus | 19530 | Vector database |
| etcd | 2379 | Milvus coordinator |
| minio | 9000 | Object storage |
| phoenix | 6006 | Trace visualization |
| bifrost | 8080 | Optional LLM gateway (profile: bifrost) |
