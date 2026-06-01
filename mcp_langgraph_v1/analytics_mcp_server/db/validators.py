from analytics_mcp_server.db.query_catalog import QUERY_CATALOG


def validate_query(query_id: str, params: dict):
    if query_id not in QUERY_CATALOG:
        raise ValueError(f"Unknown query_id: {query_id}")

    required_params = QUERY_CATALOG[query_id]["required_params"]
    missing = [
        p for p in required_params
        if p not in params or params.get(p) in [None, ""]
    ]
    if missing:
        raise ValueError(f"Missing required params: {missing}")