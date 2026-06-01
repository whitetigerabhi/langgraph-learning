-- Indexes
CREATE INDEX IF NOT EXISTS idx_batting_stats_team_season
ON cricket_mcp.batting_stats (team_id, season);

CREATE INDEX IF NOT EXISTS idx_strike_rate
ON cricket_mcp.batting_stats (strike_rate DESC);

-- View
CREATE OR REPLACE VIEW cricket_mcp.v_batting_stats_enriched AS
SELECT
    bs.season,
    t.team_code,
    p.player_name,
    bs.runs,
    bs.balls_faced,
    bs.strike_rate
FROM cricket_mcp.batting_stats bs
JOIN cricket_mcp.teams t ON t.team_id = bs.team_id
JOIN cricket_mcp.players p ON p.player_id = bs.player_id;