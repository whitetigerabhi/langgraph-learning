from pydantic import BaseModel, Field
from typing import Any

from app.dependencies import get_llm
from app.prompts.intent_prompts import INTENT_SYSTEM_PROMPT


class IntentResult(BaseModel):
    intent: str = Field(...)
    intent_confidence: float = Field(...)
    sub_intent: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    ambiguities: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)


class IntentAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(IntentResult)

    def run(self, user_query: str, normalized_query: str) -> dict:
        messages = [
            ("system", INTENT_SYSTEM_PROMPT),
            ("human", f"User query: {user_query}\nNormalized query: {normalized_query}")
        ]
        result = self.llm.invoke(messages)
        return result.model_dump()