import os
import re
from openai import AzureOpenAI
from state import AgentState
from nodes.agent_decide import TOOLS

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

def _looks_like_verbatim(answer: str, retrieved_text: str, window: int = 220) -> bool:
    """
    Simple verbatim detector:
    If any long substring (window chars) from retrieved_text is found in answer, treat as too-copy-like.
    This avoids "dumping" chunks verbatim.
    """
    if not answer or not retrieved_text:
        return False
    rt = retrieved_text.replace("\n", " ").strip()
    ans = answer.replace("\n", " ").strip()
    if len(rt) < window:
        return False
    # sample a few windows across retrieved text
    step = max(1, len(rt) // 6)
    for i in range(0, len(rt) - window, step):
        snippet = rt[i:i+window]
        if snippet in ans:
            return True
    return False

def finalize_node(state: AgentState):
    if state.get("answer"):
        return {}

    client = _client()
    messages = list(state.get("messages") or [])

    # Add a final system reminder for anti-dump synthesis.
    synthesis_guard = (
        "Now produce the final answer.\n"
        "- If you used retrieved docs, synthesize in your own words.\n"
        "- Do NOT paste long excerpts. Short quotes are allowed but must be brief.\n"
        "- Provide citations in the format [doc:chunk_id] when using retrieved material.\n"
        "- If the retrieved context does not contain the answer, say so clearly.\n"
    )

    messages2 = messages + [{"role": "system", "content": synthesis_guard}]

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages2,
        tools=TOOLS,
        tool_choice="none",
        max_tokens=450,
    )

    ans = (resp.choices[0].message.content or "").strip()

    # Attempt to detect dumping from retrieve_docs tool output if present.
    # Tool output is already in messages as JSON strings. We'll pull it out heuristically.
    retrieved_text = ""
    for m in messages:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            if '"matches"' in m["content"] and '"chunk_id"' in m["content"]:
                # best-effort extract "text" fields
                retrieved_text += " " + " ".join(re.findall(r'"text"\s*:\s*"([^"]+)"', m["content"]))

    if _looks_like_verbatim(ans, retrieved_text):
        rewrite_prompt = (
            "Rewrite the answer to avoid copying text verbatim from retrieved documents.\n"
            "- Paraphrase and summarize.\n"
            "- Keep citations [doc:chunk_id].\n"
            "- Avoid long direct quotes.\n"
            f"Original answer:\n{ans}\n"
        )
        resp2 = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages2 + [{"role": "user", "content": rewrite_prompt}],
            tools=TOOLS,
            tool_choice="none",
            max_tokens=450,
        )
        ans = (resp2.choices[0].message.content or "").strip()

    return {"answer": ans or "Done."}