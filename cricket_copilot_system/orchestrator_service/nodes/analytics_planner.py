import re
from state import CricketState


TEAM_CODES = ["CSK", "MI", "RCB", "KKR", "SRH", "RR", "DC", "PBKS", "GT", "LSG"]


def _extract_season(message: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", message)
    return int(m.group(1)) if m else None


def _extract_team(message: str) -> str | None:
    upper = message.upper()
    for t in TEAM_CODES:
        if t in upper:
            return t
    return None


def _extract_min_balls(message: str) -> int:
    # Examples:
    # "min 20 balls", "minimum 50 balls"
    m = re.search(r"(?:min(?:imum)?\s*)(\d+)\s*balls", message.lower())
    return int(m.group(1)) if m else 20


def _extract_limit(message: str) -> int:
    # "top 5", "top 10"
    m = re.search(r"top\s+(\d+)", message.lower())
    return int(m.group(1)) if m else 10


def analytics_planner_node(state: CricketState) -> CricketState:
    """
    First version: maps only a few analytics intents to query templates.
    We use deterministic extraction first for clarity and testability.
    """
    message = state["message"]
    lower = message.lower()

    season = _extract_season(message)
    team = _extract_team(message)
    min_balls = _extract_min_balls(message)
    limit = _extract_limit(message)

    if "strike rate" in lower:
        return {
            "query_id": "top_batsmen_strike_rate",
            "query_params": {
                "season": season or 2018,
                "team": team,
                "min_balls": min_balls,
                "limit": limit,
            },
        }

    if "most runs" in lower or "top scorers" in lower or "run scorers" in lower:
        return {
            "query_id": "top_run_scorers",
            "query_params": {
                "season": season or 2018,
                "team": team,
                "limit": limit,
            },
        }

    # Fallback (for this slice)
    return {
        "query_id": "top_batsmen_strike_rate",
        "query_params": {
            "season": season or 2018,
            "team": team,
            "min_balls": min_balls,
            "limit": limit,
        },
    }