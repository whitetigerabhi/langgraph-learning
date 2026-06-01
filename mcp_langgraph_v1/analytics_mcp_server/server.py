from mcp.server.fastmcp import FastMCP

from analytics_mcp_server.tools.run_stats_query import run_stats_query
from analytics_mcp_server.resources.schema_resource import get_schema_resource

mcp = FastMCP("analytics-mcp-server")


@mcp.tool()
def run_stats_query_tool(query_id: str, params_json: str) -> str:
    return run_stats_query(query_id=query_id, params_json=params_json)


@mcp.resource("db://schema")
def schema_resource() -> str:
    return get_schema_resource()


if __name__ == "__main__":
    mcp.run()