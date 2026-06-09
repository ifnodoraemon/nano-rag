# Project Context

## Repository Info
- Path: `/home/ifnodoraemon/myagent/nano-rag`
- Backend: Python 3.12 + FastAPI + Uvicorn + pytest
- Frontend: React 19 + TypeScript + Vite
- Vector DB: Milvus 2.6

## Architecture Constraints & Principles
- No frontend hardcoded business data.
- No runtime mock mode in production.
- No silent backend fallback paths.
- OpenTelemetry spans should wrap pipeline stages.
- Avoid hardcoding keyword lists.

## Milestones and Tracking Folder
All agent coordinates are in `.agents/`.
- Orchestrator: `.agents/orchestrator/`
- Subagents: `.agents/explorer_1/`, etc.
