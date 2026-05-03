import os
from openai import AzureOpenAI
from state import AgentState
from nodes.agent_decide import TOOLS

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")


def _client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )


def finalize_node(state: AgentState):
    if state.get("answer"):
        return {}

    client = _client()
    messages = list(state.get("messages") or [])

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice="none",
        max_tokens=350,
    )

    ans = (resp.choices[0].message.content or "").strip()
    return {"answer": ans or "Done."}
