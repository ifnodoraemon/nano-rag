from fastapi import APIRouter, Request

from app.schemas.document import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    container = request.app.state.container
    return await container.ingestion_pipeline.run(payload.path, kb_id=payload.kb_id or "default", tenant_id=payload.tenant_id)
