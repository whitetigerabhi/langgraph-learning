# Demo APIs (Gateway + Orchestrator)

## Gateway Service (port 8030)
POST /chat
Headers:
- Content-Type: application/json
- X-User-Role: user|admin

Body:
{
  "thread_id": "optional string",
  "message": "user text"
}

Behavior:
- RBAC + Content Safety on input
- Intent classification
- Calls orchestrator /run

## Orchestrator Service (port 8040)
POST /run
Body:
{
  "thread_id": "string",
  "query": "string",
  "user_role": "user|admin",
  "hints": { ... }
}

Behavior:
- LangGraph agentic workflow
- Tool calling (weather/cricket/retrieve_docs)
- Output Content Safety check