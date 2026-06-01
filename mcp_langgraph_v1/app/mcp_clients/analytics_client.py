import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.mcp_clients.registry import MCP_SERVER_REGISTRY
from app.utils.json_parsing import try_parse_json


class AnalyticsMCPClient:
    def __init__(self):
        cfg = MCP_SERVER_REGISTRY["analytics"]
        self.python_cmd = cfg["python_cmd"]
        self.script_path = cfg["script_path"]

    async def run_stats_query(self, query_id: str, params: dict) -> dict:
        server_params = StdioServerParameters(
            command=self.python_cmd,
            args=[self.script_path],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "run_stats_query_tool",
                    {
                        "query_id": query_id,
                        "params_json": json.dumps(params),
                    },
                )

                text = result.content[0].text
                parsed = try_parse_json(text)
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError(f"Unexpected analytics tool response: {text}")