from app.config import settings


MCP_SERVER_REGISTRY = {
    "analytics": {
        "python_cmd": settings.PYTHON_CMD,
        "script_path": settings.ANALYTICS_MCP_SERVER_SCRIPT,
    },
    "docs": {
        "python_cmd": settings.PYTHON_CMD,
        "script_path": settings.DOCS_MCP_SERVER_SCRIPT,
    },
}