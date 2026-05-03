# LangGraph Multi-Deployment Demo (Gateway + Orchestrator)# LangGraph Multi-Deployment Demo (Gateway + Or) + LLM intent classifier + routes to orchestrator
- Orchestrator API (8040): LangGraph tool-calling agent (multi-tool per step) + SQLite checkpointer memory

## Required env vars
CONTENT_SAFETY_ENDPOINT
CONTENT_SAFETY_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
CRICKETDATA_API_KEY

Optional:
CS_BLOCK_THRESHOLD (default 4)
ORCHESTRATOR_URL (default http://localhost:8040)

## Run (two terminals)

### Terminal A: Orchestrator
cd langgraph_multi_deployment_demo/orchestrator_service
mkdir -p storage
export CHECKPOINT_DB="./storage/checkpoints.sqlite"
uvicorn app:app --host 0.0.0.0 --port 8040

### Terminal B: Gateway
cd langgraph_multi_deployment_demo/gateway_service
export ORCHESTRATOR_URL="http://localhost:8040"
uvicorn app:app --host 0.0.0.0 --port 8030

## Test
THREAD=$(curl -s http://localhost:8040/threads/new | python3 -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")

curl -s -X POST http://localhost:8030/chat \
  -H 'Content-Type: application/json' \
  -H 'X-User-Role: user' \
  --data-binary "{\"thread_id\":\"$THREAD\",\"message\":\"Tell me the weather for CV32 7SU and the live cricket score for CSK.\"}"

## Services
