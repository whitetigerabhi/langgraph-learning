QUERY_CATALOG = {
    "top_batsmen_strike_rate": {
        "required_params": ["team", "season"],
        "optional_params": ["min_balls", "limit"],
    },
    "team_win_summary": {
        "required_params": ["team", "season"],
        "optional_params": [],
    },
}