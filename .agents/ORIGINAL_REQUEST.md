# Original User Request

## Initial Request — 2026-06-08T09:03:16Z

Implement 5 major "Pro-Level" architectural optimizations into the existing nano-rag codebase: Streaming JSON Parser, Metadata Pre-filtering, Parent-Child Retriever, Agentic Retry, and Query Decomposition. The goal is to elevate the RAG system to production-grade performance, robustness, and speed.

Working directory: /home/ifnodoraemon/myagent/nano-rag
Integrity mode: development

## Requirements

### R1. Streaming Output & Frontend Integration
Implement a streaming JSON parser in the backend to stream extracted answers via Server-Sent Events (SSE) at the `/chat/stream` endpoint. Update the frontend web application to consume this endpoint and render the answer with a real-time typewriter effect.

### R2. Parent-Child Retriever & Migration
Modify the chunking and vector storage logic to support a Parent-Child relationship (small chunks for embedding, large parent blocks for LLM context). Provide a complete data migration/re-ingestion script.

### R3. LLM-based Metadata Pre-filtering
Inject a lightweight LLM intent extraction step prior to retrieval. It should convert the user's query into structured metadata filters (e.g., extracting entity types or dates) to pre-filter the vector database before reranking.

### R4. Agentic Retry
Enhance the existing `corrective_recall_node` to dynamically generate alternate queries (e.g., synonyms or broader terms) when the initial retrieval returns `is_answerable: false`, enabling a second retrieval attempt.

### R5. Query Decomposition
Expand the existing `_intent_decomposition_node` to accurately break down complex multi-hop queries (e.g., "Compare X and Y") into parallel distinct retrieval sub-queries.

## Acceptance Criteria

### Streaming
- [ ] A `curl` request to `/chat/stream` returns chunks progressively rather than waiting for the entire LLM generation to finish.
- [ ] The frontend UI successfully renders the text in a typewriter fashion.

### Parent-Child Retrieval
- [ ] Running a programmatic test query demonstrates that a matched child chunk successfully yields its full parent document text in the final context.
- [ ] The provided data migration script executes without errors and successfully populates the vector store with linked parent/child records.

### Retrieval Intelligence
- [ ] Programmatic unit tests confirm that the metadata pre-filter correctly extracts standard entities (e.g., years) from raw questions.
- [ ] Mock tests confirm that the agentic retry loop triggers a secondary retrieval when the first yields no usable results.
- [ ] A test for query decomposition proves that a query asking to compare two distinct entities is split into at least two discrete retrieval queries.
