import json

from analytics_mcp_server.db.connection import get_connection
from analytics_mcp_server.db.templates import SQL_TEMPLATES
from analytics_mcp_server.db.validators import validate_query


def run_stats_query(query_id: str, params_json: str) -> str:
    params = json.loads(params_json)
    validate_query(query_id, params)

    if query_id == "top_batsmen_strike_rate":
        safe_params = {
            "team": params["team"],
            "season": params["season"],
            "min_balls": params.get("min_balls", 20),
            "limit": params.get("limit", 5),
        }
    elif query_id == "team_win_summary":
        safe_params = {
            "team": params["team"],
            "season": params["season"],
        }
    else:
        raise ValueError(f"Unsupported query_id: {query_id}")

    sql = SQL_TEMPLATES[query_id]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, safe_params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    result = {
        "query_id": query_id,
        "status": "success",
        "columns": columns,
        "rows": [dict(zip(columns, row)) for row in rows],
        "metadata": {"row_count": len(rows)},
    }
    return json.dumps(result, default=str)