-- Teams
INSERT INTO cricket_mcp.teams (team_code, team_name) VALUES
('CSK','Chennai Super Kings'),
('MI','Mumbai Indians')
ON CONFLICT DO NOTHING;

-- Players
INSERT INTO cricket_mcp.players (player_name) VALUES
('MS Dhoni'),
('Suresh Raina'),
('Shane Watson'),
('Ambati Rayudu')
ON CONFLICT DO NOTHING;

-- Batting stats for 2018 CSK
INSERT INTO cricket_mcp.batting_stats
(season, team_id, player_id, runs, balls_faced, strike_rate)
SELECT
    2018,
    t.team_id,
    p.player_id,
    s.runs,
    s.balls,
    s.sr
FROM (
    VALUES
    ('CSK','MS Dhoni', 455,302,150.66),
    ('CSK','Suresh Raina',445,320,139.06),
    ('CSK','Shane Watson',555,360,154.17),
    ('CSK','Ambati Rayudu',602,381,158.01)
) AS s(team_code, player_name, runs, balls, sr)
JOIN cricket_mcp.teams t ON t.team_code = s.team_code
JOIN cricket_mcp.players p ON p.player_name = s.player_name;
``