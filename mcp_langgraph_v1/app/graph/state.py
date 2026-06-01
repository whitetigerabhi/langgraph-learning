from typing import TypedDict, List, Dict, Any, Optional, Literal


RouteType = Literal["analytics", "retrieval", "mixed", "clarify", "unsupported"]


class MCPToolCall(TypedDict, total=False):
    server: str
    tool_name: str
    arguments: Dict[str, Any]
    status: Literal["pending", "success", "failed"]
    raw_result: Any
    error: Optional[str]


class EvidenceItem(TypedDict, total=False):
    source_type: Literal["analytics", "document", "memory"]
    source_id: str
    content: Any
    relevance_score: Optional[float]


class GraphState(TypedDict, total=False):
    # Session
    thread_id: str
    user_query: str
    normalized_query: str

    # Interpretation
    intent: RouteType
    intent_confidence: float
    sub_intent: Optional[str]
    entities: Dict[str, Any]
    ambiguities: List[str]

    # Adequacy / routing
    required_fields: List[str]
    missing_fields: List[str]
    is_adequate: bool
    route: Optional[RouteType]

    # Tool execution
    planned_tool_calls: List[MCPToolCall]
    executed_tool_calls: List[MCPToolCall]

    # Results
    analytics_result: Optional[Dict[str, Any]]
    retrieval_result: Optional[Dict[str, Any]]
    memory_result: Optional[Dict[str, Any]]

    # Evidence
    evidence: List[EvidenceItem]

    # Output
    clarification_question: Optional[str]
    answer_draft: Optional[str]
    final_answer: Optional[str]

    # Ops / debug
    warnings: List[str]
    errors: List[str]
    trace: List[str]