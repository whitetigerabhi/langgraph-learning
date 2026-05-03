import os
from openai import AzureOpenAI
from state import AgentState

def _client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location (city or UK postcode).",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cricket",
            "description": "Get live cricket/score for a team keyword (e.g., CSK).",
            "parameters": {
                "type": "object",
                "properties": {"team": {"type": "string"}},
                "required": ["team"],
                "additionalProperties": False
            },
        },
    },
]

def agent_decide_node(state: AgentState):
    if state.get("answer"):
        return {}

    step = int(state.get("step", 0))
    if step >= int(state.get("max_steps", 6)):
        return {"answer": "Max steps reached. Please rephrase your request.", "error": "max_steps"}

    hints = state.get("hints") or {}
    client = _client()

    messages = list(state.get("messages") or [])
    if not messages:
        sys = (
            "You are a tool-using agent.\n"
            "If you need current data, call tools.\n"
            "If both weather and cricket are needed, call BOTH tools in the SAME turn.\n"
            "If no tools are needed, answer directly.\n"
        )
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"{state.get('query','')}\nHints: {hints}"},
        ]

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=400,
    )

    msg = resp.choices[0].message
    tool_calls = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })

    # Persist assistant message for continuity
    messages.append({"role": "assistant", "content": msg.content or ""})

    return {"messages": messages, "tool_calls": tool_calls, "step": step + 1}