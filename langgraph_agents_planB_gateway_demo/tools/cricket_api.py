import os
import requests

CURRENT_MATCHES_URL = "https://api.cricapi.com/v1/currentMatches"

def fetch_live_cricket raise ValueError("CRICKETDATA_API_KEY not set")def fetch_live_cricket(query: str = "") -> dict:

    params = {"apikey": apikey, "offset": 0}
    r = requests.get(CURRENT_MATCHES_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Cricket API status={data.get('status')}")

    matches = data.get("data") or []
    live = [m for m in matches if m.get("matchStarted") and not m.get("matchEnded")]

    q = (query or "").strip().lower()

    def simplify(m):
        teams = m.get("teams") or []
        score = m.get("score") or []
        return {
            "teams": teams,
            "status": m.get("status", ""),
            "score": score,
            "match_name": m.get("name", "")
        }

    live_simple = [simplify(m) for m in live]

    if not live_simple:
        return {"live": [], "selected": None}

    if q:
        for m in live_simple:
            hay = (" ".join(m["teams"]) + " " + m["match_name"]).lower()
            if q in hay:
                return {"live": live_simple[:5], "selected": m}
        return {"live": live_simple[:5], "selected": None}

    return {"live": live_simple[:5], "selected": live_simple[0]}
    apikey = os.environ.get("CRICKETDATA_API_KEY", "")
    if not apikey:
