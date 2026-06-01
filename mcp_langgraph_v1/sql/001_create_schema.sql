CREATE SCHEMA IF NOT EXISTS cricket_mcp;

-- Teams
CREATE TABLE IF NOT EXISTS cricket_mcp.teams (
    team_id SERIAL PRIMARY KEY,
    team_code TEXT UNIQUE NOT NULL,
    team_name TEXT NOT NULL
);

-- Players
CREATE TABLE IF NOT EXISTS cricket_mcp.players (
    player_id SERIAL PRIMARY KEY,
    player_name TEXT UNIQUE NOT NULL,
    country TEXT
);

-- Batting stats
CREATE TABLE IF NOT EXISTS cricket_mcp.batting_stats (
    id BIGSERIAL PRIMARY KEY,
    season INT NOT NULL,
    team_id INT REFERENCES cricket_mcp.teams(team_id),
    player_id INT REFERENCES cricket_mcp.players(player_id),
    runs INT,
    balls_faced INT,
    strike_rate NUMERIC(10,2)
);

-- Bowling stats
CREATE TABLE IF NOT EXISTS cricket_mcp.bowling_stats (
    id BIGSERIAL PRIMARY KEY,
    season INT,
    team_id INT REFERENCES cricket_mcp.teams(team_id),
    player_id INT REFERENCES cricket_mcp.players(player_id),
    wickets INT,
    economy NUMERIC(10,2)
);

-- Matches
CREATE TABLE IF NOT EXISTS cricket_mcp.match_results (
    id BIGSERIAL PRIMARY KEY,
    season INT,
    team1_id INT,
    team2_id INT,
    winner_team_id INT
);