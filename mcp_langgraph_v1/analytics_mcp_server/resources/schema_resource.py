import json
import os

SCHEMA = os.environ.get("PGSCHEMA", "cricket_mcp")


def get_schema_resource() -> str:
    payload = {
        "schema": SCHEMA,
        "tables": [
            "teams",
            "players",
            "batting_stats",
            "bowling_stats",
            "match_results",
            "docs_registry",
        ],
        "views": [
            "v_batting_stats_enriched",
            "v_bowling_stats_enriched",
            "v_team_win_summary",
        ],
        "supported_query_ids": [
            "top_batsmen_strike_rate",
            "team_win_summary",
        ],
    }
    return json.dumps(payload, indent=2)