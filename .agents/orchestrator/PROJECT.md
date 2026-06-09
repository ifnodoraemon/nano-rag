# Project: nano-rag Optimizations

## Architecture
- FastAPI endpoints for Chat API (Streaming R1)
- Ingestion, chunking and vector storage with parent-child relationship (R2)
- Intent parsing & metadata pre-filtering (R3)
- Corrective Recall Node agentic retry (R4)
- Intent decomposition node query splitting (R5)

## Milestones
| # | Name | Scope | Dependencies | Status | Conv ID |
|---|------|-------|-------------|--------|---------|
| 1 | Exploration & Design | Exploration of code to target changes | None | IN_PROGRESS | 66509a44-a04a-4551-bb8f-856607fb0dad, fd623fca-de84-4284-a916-f86d07d7e355, 67d252a2-e6fc-4d63-94d8-ce80869e66c6 |
| 2 | Streaming (R1) | SSE chat stream endpoint & frontend rendering | M1 | PLANNED | |
| 3 | Parent-Child (R2) | Child vector database chunking, parent context retriever | M1 | PLANNED | |
| 4 | Pre-filtering (R3) | LLM metadata extraction + Milvus pre-filter | M1 | PLANNED | |
| 5 | Agentic Retry (R4) | corrective_recall_node synonym expansion | M1 | PLANNED | |
| 6 | Query Decomp (R5) | Compare query split in _intent_decomposition_node | M1 | PLANNED | |
| 7 | Testing & Hardening | E2E test suite + Challenger + Forensic audit | M2-M6 | PLANNED | |

## Interface Contracts
- `/chat/stream` SSE Endpoint: Progressive JSON chunk streaming.
- Parent-Child Retrieval: Matched child chunk retrieves its parent's text.
- Metadata pre-filtering: Pre-filters Milvus queries prior to retrieval using structured metadata.
- Agentic Retry: Dynamic fallback query generation if first recall fails.
- Query Decomposition: Breaks multi-hop queries down into parallel sub-queries.
