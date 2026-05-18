TEMPLATES = {
    # Top batsmen by strike rate for a season, optional team filter
    "top_batsmen_strike_rate": """
        SELECT
          player_name,
          team,
          ROUND((runs::numeric / NULLIF(balls, 0)) * 100, 2) AS strike_rate,
          balls
        FROM stats.batting_innings
        WHERE season = %(season)s
          AND balls >= %(min_balls)s
          AND (%(team)s IS NULL OR team = %(team)s)
        ORDER BY strike_rate DESC
        LIMIT %(limit)s
    """,

    # Top run scorers
    "top_run_scorers": """
        SELECT
          player_name,
          team,
          SUM(runs) AS total_runs,
          SUM(balls) AS total_balls
        FROM stats.batting_innings
        WHERE season = %(season)s
          AND (%(team)s IS NULL OR team = %(team)s)
        GROUP BY player_name, team
        ORDER BY total_runs DESC
        LIMIT %(limit)s
    """,
}