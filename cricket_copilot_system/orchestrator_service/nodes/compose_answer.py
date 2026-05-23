from state import CricketState


def compose_answer_node(state: CricketState) -> CricketState:
    result = state.get("action_result", {})
    rows = result.get("rows", [])
    cols = result.get("columns", [])
    query_id = state.get("query_id", "")

    if result.get("status") != "ok":
        return {"answer": "I could not complete the analytics request right now."}

    if not rows:
        return {"answer": "I could not find any matching records for that query."}

    if query_id == "top_batsmen_strike_rate":
        lines = ["Here are the top batsmen by strike rate:"]
        for i, row in enumerate(rows, start=1):
            player_name, team, strike_rate, balls = row
            lines.append(f"{i}. {player_name} ({team}) — strike rate {strike_rate}, balls faced {balls}")
        return {"answer": "\n".join(lines)}

    if query_id == "top_run_scorers":
        lines = ["Here are the top run scorers:"]
        for i, row in enumerate(rows, start=1):
            player_name, team, total_runs, total_balls = row
            lines.append(f"{i}. {player_name} ({team}) — runs {total_runs}, balls faced {total_balls}")
        return {"answer": "\n".join(lines)}

    return {"answer": f"I found {len(rows)} rows for your analytics query."}