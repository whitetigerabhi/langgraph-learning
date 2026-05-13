import os
from openai import AzureOpenAI
from state import AgentState

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

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
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cricket",
            "description": "Get live cricket score info for a team keyword (e.g., CSK).",
            "parameters": {
                "type": "object",
                "properties": {"team": {"type": "string"}},
                "required": ["team"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": (
                "Retrieve relevant snippets from the local knowledge base (rag_docs). "
                "Use this for questions about internal policies, runbooks, product docs, or API contracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

def agent_decide_node(state: AgentState):
    if state.get("answer"):
        return {}

    step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps", 6))
    if step >= max_steps:
        return {"answer": "Max steps reached. Please rephrase your request.", "error": "max_steps"}

    query = (state.get("query") or "").strip()
    hints = state.get("hints") or {}

    messages = list(state.get("messages") or [])

    system_prompt = (
        "You are a tool-using agent.\n"
        "\n"
        "Hard rules:\n"
        "- If the user needs current weather, call get_weather.\n"
        "- If the user needs live cricket score, call get_cricket.\n"
        "- If BOTH are needed, CALL BOTH tools in the SAME turn.\n"
        "- If the user asks about internal policy/runbook/product/API docs, call retrieve_docs.\n"
        "- Do NOT fabricate tool outputs.\n"
        "- If no tools are needed, answer directly.\n"
        "\n"
        "When you use retrieve_docs:\n"
        "- You MUST synthesize in your own words.\n"
        "- You MUST cite sources like [doc:chunk_id].\n"
        "- Do NOT paste long excerpts from the retrieved text.\n"
    )

    if not messages:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{query}\nHints: {hints}"},
        ]
    else:
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{query}\nHints: {hints}"})

    client = _client()
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=400,
    )

    msg = resp.choices[0].message

    tool_calls = []
    tool_calls_for_messages = []

    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments})
            tool_calls_for_messages.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })

    assistant_msg = {"role": "assistant", "content": msg.content or ""}
    if tool_calls_for_messages:
        assistant_msg["tool_calls"] = tool_calls_for_messages
    messages.append(assistant_msg)

    updates = {"messages": messages, "tool_calls": tool_calls, "step": step + 1}

    if not tool_calls and (msg.content or "").strip():
        updates["answer"] = (msg.content or "").strip()

    return updates