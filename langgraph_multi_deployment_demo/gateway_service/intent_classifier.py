import os
import json
import re
from typing import Dict, Any, Tuple, Optional
from openai import AzureOpenAI

# ----------------------------
# Config
# ----------------------------
# If you have a cheaper classifier deployment, set:
#   AZURE_OPENAI_CLASSIFIER_DEPLOYMENT=<your-mini-deployment>
# Otherwise it will use AZURE_OPENAI_DEPLOYMENT.
CLASSIFIER_DEPLOYMENT = (
    os.environ.get("AZURE_OPENAI_CLASSIFIER_DEPLOYMENT")
    or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "chat-model")
)

API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

# If LLM confidence is very low, we can rely on deterministic fallback
CONFIDENCE_FLOOR = float(os.environ.get("INTENT_CONFIDENCE_FLOOR", "0.35"))

# Common IPL/team tokens to detect quickly (extend if you want)
TEAM_TOKENS = {
    "CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS", "GT", "LSG"
}

UK_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.IGNORECASE)

WEATHER_KEYWORDS = (
    "weather", "temperature", "temp", "rain", "wind", "forecast", "humidity", "climate"
)

CRICKET_KEYWORDS = (
    "cricket", "ipl", "score", "live score", "match", "wicket", "overs", "run rate", "innings"
)


# ----------------------------
# Azure OpenAI client
# ----------------------------
def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=API_VERSION,
    )


# ----------------------------
# Helpers
# ----------------------------
def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction: accepts plain JSON or JSON embedded in text.
    """
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


def _normalize_intent(intent: str) -> str:
    intent = (intent or "").strip().lower()
    if intent in ("weather", "cricket", "general", "invalid", "multi"):
        return intent
    return "invalid"


def _extract_team_heuristic(text: str) -> str:
    """
    Heuristic: find any known team token in the text.
    """
    t = (text or "").upper()
    for tok in TEAM_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", t):
            return tok
    return ""


def _extract_location_heuristic(text: str) -> str:
    """
    Heuristic: try UK postcode first; otherwise return "" and let tools/LLM clarify later.
    """
    m = UK_POSTCODE_RE.search(text or "")
    if m:
        return m.group(1).upper().strip()
    return ""


def _has_weather_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in WEATHER_KEYWORDS) or bool(UK_POSTCODE_RE.search(t))


def _has_cricket_intent(text: str) -> bool:
    t = (text or "").lower()
    if any(k in t for k in CRICKET_KEYWORDS):
        return True
    # If a known team token appears, that's a strong cricket signal
    return bool(_extract_team_heuristic(text))


def _deterministic_fallback(text: str) -> Dict[str, Any]:
    """
    Enterprise-friendly deterministic router:
    - Detect weather/cricket/multi from keywords + regex
    - Extract team/postcode hints if present
    """
    has_w = _has_weather_intent(text)
    has_c = _has_cricket_intent(text)

    if has_w and has_c:
        intent = "multi"
        conf = 0.60
    elif has_w:
        intent = "weather"
        conf = 0.65
    elif has_c:
        intent = "cricket"
        conf = 0.65
    else:
        # Many general knowledge questions won't include our keywords.
        # If it's non-empty, treat as "general" rather than invalid.
        intent = "general" if (text or "").strip() else "invalid"
        conf = 0.45 if intent == "general" else 0.0

    return {
        "intent": intent,
        "location": _extract_location_heuristic(text),
        "team": _extract_team_heuristic(text),
        "confidence": conf,
        "source": "deterministic",
    }


def _llm_classify(text: str) -> Dict[str, Any]:
    """
    LLM classifier: returns intent + optional hints.
    """
    client = _client()

    prompt = f"""
You are an intent classifier for a gateway.

Return ONLY JSON with this exact schema:
{{
  "intent": "weather|cricket|general|multi|invalid",
  "location": "",
  "team": "",
  "confidence": 0.0
}}

Rules:
- "weather": user asks about current weather/forecast for a place or postcode.
- "cricket": user asks about live cricket match/score, IPL, wickets, teams etc.
- "general": general knowledge questions not needing our tools (e.g., universe size, dog lifespan).
- "multi": request clearly includes BOTH weather and cricket in one message.
- "invalid": empty or meaningless request.

Hints:
- If you see a UK postcode like "CV32 7SU", put it in "location".
- If you see a team like "CSK", put it in "team".
- Confidence is 0 to 1.

User message:
{text}
""".strip()

    resp = client.chat.completions.create(
        model=CLASSIFIER_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return only JSON. No markdown, no prose."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=180,
    )

    raw = resp.choices[0].message.content or ""
    data = _extract_json(raw)

    intent = _normalize_intent(data.get("intent", "invalid"))
    location = (data.get("location") or "").strip()
    team = (data.get("team") or "").strip()
    try:
        confidence = float(data.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0

    return {
        "intent": intent,
        "location": location,
        "team": team,
        "confidence": confidence,
        "source": "llm",
    }


# ----------------------------
# Public API
# ----------------------------
def classify_intent(message: str) -> Dict[str, Any]:
    """
    Main classifier used by gateway_service/app.py.

    Strategy:
    1) Try LLM classification (more flexible).
    2) If result is 'invalid' or confidence is low, use deterministic fallback.
    3) Merge heuristic extraction for missing team/location.
    """
    text = (message or "").strip()

    # Empty message => invalid immediately
    if not text:
        return {"intent": "invalid", "location": "", "team": "", "confidence": 0.0, "source": "deterministic"}

    llm = _llm_classify(text)

    # If LLM is unsure or says invalid, fallback to deterministic.
    if llm["intent"] == "invalid" or llm["confidence"] < CONFIDENCE_FLOOR:
        det = _deterministic_fallback(text)

        # If deterministic found strong signals, prefer it; otherwise keep LLM (e.g., general questions)
        # Rule: deterministic wins for weather/cricket/multi because those drive tooling.
        if det["intent"] in ("weather", "cricket", "multi"):
            chosen = det
        else:
            chosen = llm
    else:
        chosen = llm

    # Fill missing hints with heuristics (non-invasive)
    if not chosen.get("location"):
        chosen["location"] = _extract_location_heuristic(text)
    if not chosen.get("team"):
        chosen["team"] = _extract_team_heuristic(text)

    # Final normalization
    chosen["intent"] = _normalize_intent(chosen.get("intent"))
    if "confidence" not in chosen:
        chosen["confidence"] = 0.0

    return chosen