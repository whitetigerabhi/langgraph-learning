import os
from openai import AzureOpenAI
from state import AgentState

# Use your deployed Azure OpenAI deployment name
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")


def _client() -> AzureOpenAI:
    """
    Azure OpenAI client using API key auth.
    """
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )


# Tools exposed to the model (LLM-native tool calling)
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
]


def _needs_new_user_message(messages: list, query: str) -> bool:
    """
    If the conversation already has messages, ensure the current query is actually present
    as the latest user turn. This helps avoid the agent "sticking" to an old topic when
    you reuse a thread_id for multiple user requests.
    """
    if not messages:
        return True
    # Find last user message content
    last_user = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
            break
    return (last_user != (query or "").strip())


def agent_decide_node(state: AgentState):
    """
    Non-deterministic agent step:
    - Calls Azure OpenAI with tool schemas enabled
    - Model may return multiple tool calls in one step (parallel tool calling)
    - We store tool calls in state.tool_calls (simple dict list)
    - We also append an assistant message with tool_calls in state.messages (OpenAI format)

    If model returns no tool_calls and provides a direct answer, we set state.answer directly.
    """
    # If we already have an answer (or earlier node blocked), do nothing.
    if state.get("answer"):
        return {}

    step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps", 6))
    if step >= max_steps:
        return {
            "answer": "Max steps reached. Please rephrase your request.",
            "error": "max_steps",
        }

    query = (state.get("query") or "").strip()
    hints = state.get("hints") or {}

    client = _client()

    # Build/extend conversation messages
    messages = list(state.get("messages") or [])

    system_prompt = (
        "You are a tool-using agent.\n"
        "\n"
        "Hard rules:\n"
        "- If the user needs current weather information, call get_weather.\n"
        "- If the user needs live cricket score information, call get_cricket.\n"
        "- If BOTH are needed, CALL BOTH tools in the SAME turn.\n"
        "- Do NOT fabricate tool outputs.\n"
        "- If no tool is needed, answer directly.\n"
    )

    if not messages:
        # First turn in this thread
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{query}\nHints: {hints}"},
        ]
    else:
        # Ensure the current query is present as the latest user message
        # (Prevents reusing an old query context unintentionally)
        if _needs_new_user_message(messages, query):
            # Keep original system message if present; otherwise insert
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": f"{query}\nHints: {hints}"})

    # Call model with tool support
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=400,
    )

    msg = resp.choices[0].message

    # Normalize tool calls into JSON-serializable list
    tool_calls = []
    tool_calls_for_messages = []

    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
            # Keep OpenAI-style tool_calls on the assistant message
            tool_calls_for_messages.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

    # Append assistant message, including tool_calls if present
    assistant_msg = {"role": "assistant", "content": msg.content or ""}
    if tool_calls_for_messages:
        assistant_msg["tool_calls"] = tool_calls_for_messages
    messages.append(assistant_msg)

    updates = {
        "messages": messages,
        "tool_calls": tool_calls,
        "step": step + 1,
    }

    # If no tools were requested and model gave a direct answer, store it now.
    # finalize_node() will then no-op (because answer already exists).
    if not tool_calls and (msg.content or "").strip():
        updates["answer"] = (msg.content or "").strip()

    return updates