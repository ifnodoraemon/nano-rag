# Implementation Plan: 5 Major "Pro-Level" Architectural Optimizations

## Milestones

### Milestone 1: Initial Exploration and Technical Design
- **Objective**: Explore the codebase, identify key integration points for each requirement, and produce a detailed implementation strategy.
- **Verification**: Explorer handoff reports highlighting files to change.

### Milestone 2: Streaming Output & Frontend Integration (R1)
- **Objective**: Implement a streaming JSON parser at `/chat/stream` (SSE) and typewriter effect on the frontend.
- **Verification**: `curl` to `/chat/stream` showing streaming chunks; frontend builds and renders correctly.

### Milestone 3: Parent-Child Retriever & Migration (R2)
- **Objective**: Create parent-child chunking & vector storage + migration script.
- **Verification**: Run data migration script, verify child retrieval maps to parent context.

### Milestone 4: Retrieval Intelligence: LLM-based Metadata Pre-filtering (R3)
- **Objective**: Extract structured metadata filter (e.g. years) before retrieval.
- **Verification**: Unit tests showing correct extraction and pre-filtering on Milvus.

### Milestone 5: Retrieval Intelligence: Agentic Retry (R4)
- **Objective**: Support secondary query generation and retrieval when first attempt is not answerable.
- **Verification**: Unit tests verifying `corrective_recall_node` triggers a retry query.

### Milestone 6: Retrieval Intelligence: Query Decomposition (R5)
- **Objective**: Split multi-hop queries into multiple parallel sub-queries in `_intent_decomposition_node`.
- **Verification**: Unit tests verifying a comparison query is split into at least two discrete sub-queries.

### Milestone 7: Dual Track: E2E Integration and Adversarial Hardening (Tiers 1-5)
- **Objective**: Integration testing, full suite execution, challenger testing, forensic audit, and final verification.
- **Verification**: All tests passing, 0 lints, clean forensic audit.

---

## Verification Protocols
- **Unit/Integration Tests**: Standard pytest suite for each feature.
- **E2E Testing**: Dual-track requirement-driven opaque-box testing.
- **Adversarial Hardening**: Challengers testing code for edge cases and gaps.
- **Forensic Audit**: Auditing codebase for integrity and correctness.
