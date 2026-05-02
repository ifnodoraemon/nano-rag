from pydantic import BaseModel, Field


class DiagnosisFinding(BaseModel):
    category: str
    severity: str
    rationale: str
    suggested_actions: list[str] = Field(default_factory=list)
    evidence: dict[str, object] = Field(default_factory=dict)


class DiagnosisResponse(BaseModel):
    target_type: str
    trace_id: str | None = None
    sample_id: str | None = None
    summary: str
    findings: list[DiagnosisFinding] = Field(default_factory=list)
    ai_suggestion: str | None = None


class TraceDiagnosisRequest(BaseModel):
    trace_id: str
    session_id: str | None = None
    include_ai: bool = False


class EvalDiagnosisRequest(BaseModel):
    report_path: str
    result_index: int
    include_ai: bool = False


class AutoDiagnosisRequest(BaseModel):
    include_ai: bool = True
