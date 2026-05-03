import os, json, re
from openai import AzureOpenAI

_CLASSIFIER_DEPLOYMENT") or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")


def _client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-10-21",
    )


def _extract_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def classify_intent(message: str) -> dict:
    client = _client()
    prompt = f"""
Return ONLY JSON:
{{
  "intent":"weather|cricket|general|invalid",
  "location":"",
  "team":"",
  "confidence":0.0
}}
User message: {message}
"""
    resp = client.chat.completions.create(
        model=CLASSIFIER_DEPLOYMENT,
        messages=[{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": prompt}],
        max_tokens=200,
    )
    data = _extract_json(resp.choices[0].message.content.strip())
    data["intent"] = (data.get("intent") or "invalid").strip()
    data["location"] = (data.get("location") or "").strip()
    data["team"] = (data.get("team") or "").strip()
    try:
        data["confidence"] = float(data.get("confidence") or 0.0)
    except Exception:
        data["confidence"] = 0.0
    return data
