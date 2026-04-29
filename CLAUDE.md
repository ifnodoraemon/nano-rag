# nano-rag

Enterprise RAG system with a real-data runtime: no frontend hardcoded business data, no runtime mock mode, and no silent backend fallback paths.

## Principles

- Real backend data is the source of truth for workspaces, ingest sources, documents, traces, eval reports, and diagnosis targets.
- Runtime failures should be visible. Missing provider keys, invalid model credentials, parser errors, and Milvus errors must not be hidden behind mock or fallback behavior.
- The frontend does not store or submit business API keys. Browser requests go through the account system or the Docker nginx proxy, which injects the local backend key.
- Every RAG answer should be traceable to retrieved context, citations, source documents, and trace IDs.

## Tech Stack

- **Backend**: Python 3.12 + FastAPI + Uvicorn
- **Frontend**: React 19 + TypeScript + Vite, served by nginx in Docker
- **Vector DB**: Milvus 2.6 standalone with etcd + MinIO
- **Model Providers**: direct Gemini or Qwen provider configuration; no Bifrost/LiteLLM runtime layer
- **Tracing**: OpenTelemetry HTTP export to Langfuse
- **Testing**: pytest + pytest-asyncio, TypeScript build/lint
- **Evaluation**: deterministic eval plus optional RAGAS library metrics

## Project Structure

```text
app/
  api/          # FastAPI routes (business + debug)
  core/         # Config, exceptions, logging, tracing
  generation/   # Prompt builder, answer formatter, LLM service
  ingestion/    # Document parsing, chunking, metadata extraction
  model_client/ # Embedding, generation, rerank, document parser clients
  retrieval/    # Hybrid retrieval, reranking, context builder, freshness
  schemas/      # Pydantic models
  vectorstore/  # Milvus plus in-memory test repository
configs/        # settings.yaml, models.yaml, prompts.yaml
frontend/       # React SPA source
docker/         # docker-compose.yml + Dockerfiles + nginx config
scripts/        # run_eval.py, run_benchmark.py
```

## Runtime Architecture

### Ingestion Pipeline

Document -> configured parser -> normalizer -> chunker -> metadata -> multimodal embedding -> Milvus

### Retrieval Pipeline

Query -> optional query rewrite -> hybrid retrieval -> reranking -> freshness filter -> context builder

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
- `EMBEDDING_API_BASE_URL` - direct embedding provider endpoint
- `EMBEDDING_API_KEY` - direct embedding provider key
- `EMBEDDING_PROVIDER` - `gemini`, `dashscope`, or `vllm`
- `DOCUMENT_PARSER_API_BASE_URL` - document parser endpoint
- `DOCUMENT_PARSER_API_KEY` - document parser key
- `DOCUMENT_PARSER_PROVIDER` - `gemini` or `qwen`; `qwen` covers DashScope and OpenAI-compatible vLLM
- `VECTORSTORE_BACKEND` - defaults to `milvus`; `memory` is only for explicit tests/experiments
- `RAG_API_KEYS` - backend business API keys

## Code Conventions

- Pydantic v2 for schemas (`model_validate`, not `parse_obj`)
- Async FastAPI/httpx paths
- `AppContainer` owns dependency wiring in lifespan
- YAML config supports env substitution
- OpenTelemetry spans should wrap pipeline stages
- Do not introduce frontend hardcoded business data or backend fallback behavior

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 | FastAPI backend |
| frontend | 3000 | nginx + React SPA |
| langfuse | 3001 | Trace, eval, and prompt observability |
| milvus | 19530 | Vector database |
| etcd | 2379 | Milvus coordinator |
| minio | 9000 | Object storage |
