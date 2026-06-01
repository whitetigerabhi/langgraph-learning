from app.dependencies import get_llm
from app.prompts.clarify_prompts import CLARIFY_SYSTEM_PROMPT


class ClarifyAgent:
    def __init__(self):
        self.llm = get_llm()

    def run(self, user_query: str, missing_fields: list, ambiguities: list) -> str:
        messages = [
            ("system", CLARIFY_SYSTEM_PROMPT),
            ("human", f"""
User query: {user_query}
Missing fields: {missing_fields}
Ambiguities: {ambiguities}
Ask one concise clarification question.
""")
        ]
        return self.llm.invoke(messages).content