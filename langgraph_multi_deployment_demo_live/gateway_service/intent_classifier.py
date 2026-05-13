import os
import json
import re
from typing import Dict, Any
from openai import AzureOpenAI

CLASSIFIER_DEPLOYMENT = (
    os.environ.get("AZURE_OPENAI_CLASSIFIER_DEPLOYMENT")
    or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")
)

API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

WEATHER_KEYWORDS = ("weather", "temperature", "forecast", "rain", "wind", "humidity")
CRICKET_KEYWORDS = ("cricket", "ipl", "score", "live score", "wicket", "overs", "innings")
RAG_KEYWORDS = ("policy", "runbook", "internal", "guidelines", "api contract", "reimbursement", "checklist", "procedure", "kb", "knowledge base")

TEAM_TOKENS = {"CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS", "GT", "LSG"}
UK_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.IGNORECASE)

def _client():
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=API_VERSION,
    )

def _extract_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

def _team_heuristic(text: str) -> str:
    t = (text or "").upper()
    for tok in TEAM_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", t):
            return tok
    return ""

def _location_heuristic(text: str) -> str:
    m = UK_POSTCODE_RE.search(text or "")
    return (m.group(1).upper().strip() if m else "")

def _deterministic_intent(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    has_weather = any(k in t for k in WEATHER_KEYWORDS) or bool(UK_POSTCODE_RE.search(t))
    has_cricket = any(k in t for k in CRICKET_KEYWORDS) or bool(_team_heuristic(text))
    has_rag = any(k in t for k in RAG_KEYWORDS)

    if has_weather and has_cricket:
        intent = "multi"
        conf = 0.7
    elif has_weather:
        intent = "weather"
        conf = 0.7
    elif has_cricket:
        intent = "cricket"
        conf = 0.7
    elif has_rag:
        intent = "general"   # let orchestrator decide retrieval
        conf = 0.65
    else:
        intent = "general"
        conf = 0.5

    return {
        "intent": intent,
        "location": _location_heuristic(text),
        "team": _team_heuristic(text),
        "confidence": conf,
        "source": "deterministic"
    }

def classify_intent(message: str) -> Dict[str, Any]:
    text = (message or "").strip()
    if not text:
        return {"intent": "invalid", "location": "", "team": "", "confidence": 0.0, "source": "deterministic"}

    # deterministic first (enterprise-friendly + fast)
    det = _deterministic_intent(text)

    # LLM classifier as refinement (optional). If it breaks, we fall back to deterministic.
    try:
        prompt = f"""
Return ONLY JSON:
{{
  "intent":"weather|cricket|general|multi|invalid",
  "location":"",
  "team":"",
  "confidence":0.0
}}
User message: {text}
"""
        resp = _client().chat.completions.create(
            model=CLASSIFIER_DEPLOYMENT,
            messages=[{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": prompt}],
            max_tokens=200,
        )
        llm = _extract_json(resp.choices[0].message.content or "")
        intent = (llm.get("intent") or det["intent"]).strip().lower()
        if intent not in ("weather", "cricket", "general", "multi", "invalid"):
            intent = det["intent"]

        return {
            "intent": intent,
            "location": (llm.get("location") or det["location"]).strip(),
            "team": (llm.get("team") or det["team"]).strip(),
            "confidence": float(llm.get("confidence") or det["confidence"]),
            "source": "llm"
        }
    except Exception:
        return det
