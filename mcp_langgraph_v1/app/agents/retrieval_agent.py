from pydantic import BaseModel, Field
from typing import Any

from app.dependencies import get_llm
from app.prompts.retrieval_prompts import RETRIEVAL_SYSTEM_PROMPT


class RetrievalPlan(BaseModel):
    search_query: str = Field(...)
    top_k: int = Field(default=5)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(RetrievalPlan)

    def plan(self, user_query: str, entities: dict) -> dict:
        messages = [
            ("system", RETRIEVAL_SYSTEM_PROMPT),
            ("human", f"User query: {user_query}\nEntities: {entities}")
        ]
        result = self.llm.invoke(messages)
        return result.model_dump()