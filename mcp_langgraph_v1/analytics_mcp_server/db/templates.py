import os

SCHEMA = os.environ.get("PGSCHEMA", "cricket_mcp")

SQL_TEMPLATES = {
    "top_batsmen_strike_rate": f"""
        SELECT
            player_name AS player,
            strike_rate,
            balls_faced,
            runs
        FROM {SCHEMA}.v_batting_stats_enriched
        WHERE team_code = %(team)s
          AND season = %(season)s
          AND balls_faced >= %(min_balls)s
        ORDER BY strike_rate DESC, runs DESC
        LIMIT %(limit)s
    """,
    "team_win_summary": f"""
        SELECT
            season,
            team_code,
            team_name,
            wins
        FROM {SCHEMA}.v_team_win_summary
        WHERE team_code = %(team)s
          AND season = %(season)s
    """
}