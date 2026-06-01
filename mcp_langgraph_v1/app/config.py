import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Azure OpenAI
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

    # App
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8040"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # DB
    PGHOST = os.getenv("PGHOST", "")
    PGDATABASE = os.getenv("PGDATABASE", "postgres")
    PGUSER = os.getenv("PGUSER", "")
    PGPASSWORD = os.getenv("PGPASSWORD", "")
    PGPORT = int(os.getenv("PGPORT", "5432"))
    PGSSLMODE = os.getenv("PGSSLMODE", "require")
    PGSCHEMA = os.getenv("PGSCHEMA", "cricket_mcp")

    # MCP server scripts
    PYTHON_CMD = os.getenv("PYTHON_CMD", "python")
    ANALYTICS_MCP_SERVER_SCRIPT = os.getenv("ANALYTICS_MCP_SERVER_SCRIPT", "analytics_mcp_server/server.py")
    DOCS_MCP_SERVER_SCRIPT = os.getenv("DOCS_MCP_SERVER_SCRIPT", "docs_mcp_server/server.py")


settings = Settings()