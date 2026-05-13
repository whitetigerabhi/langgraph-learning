import os
import hashlib
from typing import List, Dict, Any, Tuple

import psycopg2
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# -----------------------
# Env / Config
# -----------------------
PG_HOST = os.environ["PG_HOST"]          # e.g. postgres-abhi.postgres.database.azure.com
PG_DB = os.environ.get("PG_DB", "ragdb")
PG_USER = os.environ["PG_USER"]          # e.g. abhi_admin
PG_PASSWORD = os.environ["PG_PASSWORD"]

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]  # e.g. embedding-model

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))        # chars
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))  # chars
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))

# -----------------------
# Clients
# -----------------------
aoai = AzureOpenAI(
    api_key=AOAI_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)

search = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY),
)

# -----------------------
# Helpers
# -----------------------
def chunk_text(text: str, size: int, overlap: int) -> Listtext = (text or "").replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def stable_id(prefix: str, source_id: str, chunk_index: int) -> str:
    # Stable deterministic doc id so re-ingestion updates same docs
    raw = f"{prefix}:{source_id}:{chunk_index}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{source_id}_{chunk_index}_{h}"

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]

def fetch_rows(conn) -> Tuple[List[tuple], List[tuple]]:
    cur = conn.cursor()

    # policies: (id, policy_type, title, body, effective_date, owner_team)
    cur.execute("""
        SELECT
          p.id::text,
          p.policy_type,
          p.title,
          p.body,
          COALESCE(p.effective_date::text, '') as effective_date,
          COALESCE(t.name, '') as owner_team
        FROM policies p
        LEFT JOIN teams t ON t.id = p.owner_team_id
    """)
    policies = cur.fetchall()

    # runbooks: (id, severity, title, description, steps, owner_team)
    cur.execute("""
        SELECT
          r.id::text,
          r.severity,
          r.title,
          COALESCE(r.description, '') as description,
          r.steps,
          COALESCE(t.name, '') as owner_team
        FROM incident_runbooks r
        LEFT JOIN teams t ON t.id = r.owner_team_id
    """)
    runbooks = cur.fetchall()

    cur.close()
    return policies, runbooks

def make_policy_content(row: tuple) -> Dict[str, Any]:
    source_id, policy_type, title, body, effective_date, owner_team = row
    header = (
        f"Policy\n"
        f"Policy Type: {policy_type}\n"
        f"Title: {title}\n"
        f"Effective Date: {effective_date}\n"
        f"Owner Team: {owner_team}\n"
    ).strip()
    content = f"{header}\n\n{body}".strip()
    return {
        "entity_type": "policy",
        "source_table": "policies",
        "source_id": source_id,
        "title": title,
        "content": content,
        "policy_type": policy_type,
        "severity": "",
        "owner_team": owner_team,
        "effective_date": effective_date,
    }

def make_runbook_content(row: tuple) -> Dict[str, Any]:
    source_id, severity, title, description, steps, owner_team = row
    header = (
        f"Runbook\n"
        f"Severity: {severity}\n"
        f"Title: {title}\n"
        f"Owner Team: {owner_team}\n"
    ).strip()
    body = f"{description}\n\nSteps:\n{steps}".strip()
    content = f"{header}\n\n{body}".strip()
    return {
        "entity_type": "runbook",
        "source_table": "incident_runbooks",
        "source_id": source_id,
        "title": title,
        "content": content,
        "policy_type": "",
        "severity": severity,
        "owner_team": owner_team,
        "effective_date": "",
    }

def upload_docs(docs: List[Dict[str, Any]]):
    if not docs:
        print("No docs to upload.")
        return
    results = search.merge_or_upload_documents(documents=docs)
    failed = [r for r in results if not r.succeeded]
    if failed:
        print(f"⚠️ Upload completed with {len(failed)} failures. First few:")
        print(failed[:3])
    else:
        print(f"✅ Uploaded {len(docs)} documents")

# -----------------------
# Main
# -----------------------
def main():
    conn = psycopg2.connect(
        host=PG_HOST,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        sslmode="require",
    )

    policies_rows, runbooks_rows = fetch_rows(conn)
    print(f"Fetched policies={len(policies_rows)}, runbooks={len(runbooks_rows)} from PostgreSQL")

    # Convert rows into chunkable “knowledge units”
    units: List[Dict[str, Any]] = []
    for r in policies_rows:
        units.append(make_policy_content(r))
    for r in runbooks_rows:
        units.append(make_runbook_content(r))

    # Chunk + embed + upload
    batch_docs: List[Dict[str, Any]] = []
    texts_to_embed: List[str] = []
    metas: List[Dict[str, Any]] = []

    for unit in units:
        chunks = chunk_text(unit["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            doc_id = stable_id(unit["entity_type"], unit["source_id"], i)
            metas.append((unit, i, doc_id, ch))
            texts_to_embed.append(ch)

            # embed in batches
            if len(texts_to_embed) >= EMBED_BATCH_SIZE:
                embs = embed_texts(texts_to_embed)
                for emb, (u, idx, did, text) in zip(embs, metas):
                    batch_docs.append({
                        "id": did,
                        "entity_type": u["entity_type"],
                        "title": u["title"],
                        "content": text,

                        "source_table": u["source_table"],
                        "source_id": u["source_id"],
                        "chunk_index": idx,

                        "policy_type": u["policy_type"],
                        "severity": u["severity"],
                        "owner_team": u["owner_team"],
                        "effective_date": u["effective_date"],

                        "embedding": emb,
                    })
                texts_to_embed, metas = [], []

                # upload periodically to keep memory low
                if len(batch_docs) >= 500:
                    upload_docs(batch_docs)
                    batch_docs = []

    # flush remainder
    if texts_to_embed:
        embs = embed_texts(texts_to_embed)
        for emb, (u, idx, did, text) in zip(embs, metas):
            batch_docs.append({
                "id": did,
                "entity_type": u["entity_type"],
                "title": u["title"],
                "content": text,

                "source_table": u["source_table"],
                "source_id": u["source_id"],
                "chunk_index": idx,

                "policy_type": u["policy_type"],
                "severity": u["severity"],
                "owner_team": u["owner_team"],
                "effective_date": u["effective_date"],

                "embedding": emb,
            })

    if batch_docs:
        upload_docs(batch_docs)

    conn.close()
    print("✅ Done")

if __name__ == "__main__":
    main()