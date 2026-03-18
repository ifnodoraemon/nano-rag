from pydantic import BaseModel


class RerankResult(BaseModel):
    index: int
    score: float
    document: str
