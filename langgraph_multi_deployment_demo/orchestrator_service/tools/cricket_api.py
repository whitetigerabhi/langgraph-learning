import os
import requests

# CricAPI current matches endpoint (CricAPI v1)
CURRENT_MATCHES_URL = "https://api.cricapi.com/v1/currentMatches"


def fetch_live_cricket(query: str = "") -> dict:
    """
    Fetch current live matches from CricAPI and optionally select the match
    whose teams/name best match the query (e.g., 'CSK').

    Requires:
      - env var CRICKETDATA_API_KEY
    """
    apikey = os.environ.get("CRICKETDATA_API_KEY", "").strip()
    if not apikey:
        raise ValueError("CRICKETDATA_API_KEY not set")

    params = {"apikey": apikey, "offset": 0}
    r = requests.get(CURRENT_MATCHES_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Cricket API status={data.get('status')}")

    matches = data.get("data") or []
    live = [m for m in matches if m.get("matchStarted") and not m.get("matchEnded")]

    def simplify(m):
        teams = m.get("teams") or []
        score = m.get("score") or []
        return {
            "teams": teams,
            "status": m.get("status", ""),
            "score": score,
            "match_name": m.get("name", ""),
        }

    live_simple = [simplify(m) for m in live]
    q = (query or "").strip().lower()

    if not live_simple:
        return {"live": [], "selected": None}

    # If query provided, try to pick the best match
    if q:
        for m in live_simple:
            hay = (" ".join(m["teams"]) + " " + m["match_name"]).lower()
            if q in hay:
                return {"live": live_simple[:5], "selected": m}
        # No match found, still return a small list of live matches
        return {"live": live_simple[:5], "selected": None}

    # If no query, pick the first live match as selected
    return {"live": live_simple[:5], "selected": live_simple[0]}