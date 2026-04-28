from pydantic import BaseModel


class EvalRunRequest(BaseModel):
    dataset_path: str
    output_path: str | None = None
    use_ragas_lib: bool = False


class EvalRunResponse(BaseModel):
    status: str
    output_path: str | None = None
    report: dict
