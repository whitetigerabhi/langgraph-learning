import json
from app.dependencies import get_llm
from app.prompts.synthesis_prompts import SYNTHESIS_SYSTEM_PROMPT


class SynthesisAgent:
    def __init__(self):
        self.llm = get_llm()

    def run(self, user_query: str, route: str, evidence: list, analytics_result=None, retrieval_result=None) -> str:
        payload = {
            "route": route,
            "analytics_result": analytics_result,
            "retrieval_result": retrieval_result,
            "evidence": evidence,
        }

        messages = [
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", f"User query: {user_query}\n\nEvidence:\n{json.dumps(payload, indent=2, default=str)}")
        ]
        return self.llm.invoke(messages).content