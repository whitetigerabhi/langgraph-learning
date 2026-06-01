from pydantic import BaseModel
from typing import Any


class AnalyticsResult(BaseModel):
    query_id: str
    status: str
    columns: list[str]
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


class DocsSearchResult(BaseModel):
    matches: list[dict[str, Any]]