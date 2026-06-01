from mcp.server.fastmcp import FastMCP
from docs_mcp_server.tools.search_docs import search_docs

mcp = FastMCP("docs-mcp-server")


@mcp.tool()
def search_docs_tool(query: str, top_k: int = 5, filters_json: str = "{}") -> str:
    return search_docs(query=query, top_k=top_k, filters_json=filters_json)


if __name__ == "__main__":
    mcp.run()