from pydantic import BaseModel, Field
from typing import Any

from app.dependencies import get_llm
from app.prompts.analytics_prompts import ANALYTICS_SYSTEM_PROMPT


class AnalyticsPlan(BaseModel):
    query_id: str = Field(...)
    params: dict[str, Any] = Field(default_factory=dict)


class AnalyticsAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(AnalyticsPlan)

    def plan(self, user_query: str, entities: dict, sub_intent: str | None) -> dict:
        messages = [
            ("system", ANALYTICS_SYSTEM_PROMPT),
            ("human", f"""
User query: {user_query}
Entities: {entities}
Sub-intent: {sub_intent}
Allowed query_ids:
- top_batsmen_strike_rate
- team_win_summary
""")
        ]
        result = self.llm.invoke(messages)
        return result.model_dump()