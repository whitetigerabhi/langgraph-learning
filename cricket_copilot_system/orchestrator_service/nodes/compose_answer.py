from state import CricketState


def compose_answer_node(state: CricketState) -> CricketState:
    route = state.get("route")

    # -----------------------------
    # Trivia / retrieval path
    # -----------------------------
    if route == "trivia":
        retrieval = state.get("retrieval", {})
        matches = retrieval.get("matches", [])
        citations = retrieval.get("citations", [])

        if not matches:
            return {"answer": "I could not find relevant cricket knowledge for that question."}

        lines = ["Here’s what I found:"]
        for i, m in enumerate(matches[:3], start=1):
            title = m.get("title", "Untitled")
            content = (m.get("content") or "").strip()
            if len(content) > 220:
                content = content[:220] + "..."
            source_id = m.get("source_id", "")
            chunk_id = m.get("chunk_id", "")
            lines.append(f"{i}. {title}: {content} [{source_id}:{chunk_id}]")

        return {"answer": "\n".join(lines)}

    # -----------------------------
    # Analytics path
    # -----------------------------
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