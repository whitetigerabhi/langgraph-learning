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
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index-live")

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
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Chunk a long string into overlapping character windows.
    Overlap helps avoid cutting key phrases at boundaries.
    """
    text = (text or "").replace("\r\n", "\n")
    chunks: List[str] = []
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
    """
    Stable deterministic id so re-ingestion updates the same search docs.
    Example: policy_42_0_<hash>
    """
    raw = f"{prefix}:{source_id}:{chunk_index}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{source_id}_{chunk_index}_{h}"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch embed using Azure OpenAI embeddings deployment.
    """
    resp = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]


def fetch_rows(conn) -> Tuple[List[tuple], List[tuple]]:
    """
    Fetch policies and runbooks. Joins teams table if it exists.
    If teams doesn't exist, owner_team becomes empty string.
    """
    cur = conn.cursor()

    # Policies
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

    # Runbooks
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


def make_policy_unit(row: tuple) -> Dict[str, Any]:
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
        "full_content": content,

        # metadata fields (some will be empty for non-policy docs)
        "policy_type": policy_type,
        "severity": "",
        "owner_team": owner_team,
        "effective_date": effective_date,
    }


def make_runbook_unit(row: tuple) -> Dict[str, Any]:
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
        "full_content": content,

        "policy_type": "",
        "severity": severity,
        "owner_team": owner_team,
        "effective_date": "",
    }


def upload_docs(docs: List[Dict[str, Any]]):
    if not docs:
        return
    results = search.merge_or_upload_documents(documents=docs)
    failed = [r for r in results if not r.succeeded]
    if failed:
        print(f"⚠️ Upload completed with {len(failed)} failures. First few:")
        print(failed[:3])
    else:
        print(f"✅ Uploaded {len(docs)} documents to index '{INDEX_NAME}'")


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

    units: List[Dict[str, Any]] = []
    units.extend(make_policy_unit(r) for r in policies_rows)
    units.extend(make_runbook_unit(r) for r in runbooks_rows)

    # We'll embed and upload in batches to avoid large memory use
    batch_texts: List[str] = []
    batch_meta: List[tuple] = []
    batch_docs: List[Dict[str, Any]] = []

    for unit in units:
        chunks = chunk_text(unit["full_content"], CHUNK_SIZE, CHUNK_OVERLAP)

        for ci, chunk in enumerate(chunks):
            doc_id = stable_id(unit["entity_type"], unit["source_id"], ci)
            batch_texts.append(chunk)
            batch_meta.append((unit, ci, doc_id, chunk))

            if len(batch_texts) >= EMBED_BATCH_SIZE:
                embs = embed_texts(batch_texts)

                for emb, (u, chunk_index, did, ctext) in zip(embs, batch_meta):
                    batch_docs.append({
                        "id": did,
                        "entity_type": u["entity_type"],
                        "title": u["title"],
                        "content": ctext,

                        "source_table": u["source_table"],
                        "source_id": u["source_id"],
                        "chunk_index": chunk_index,

                        "policy_type": u["policy_type"],
                        "severity": u["severity"],
                        "owner_team": u["owner_team"],
                        "effective_date": u["effective_date"],

                        "embedding": emb,
                    })

                batch_texts, batch_meta = [], []

                # Upload periodically
                if len(batch_docs) >= 500:
                    upload_docs(batch_docs)
                    batch_docs = []

    # Flush remaining
    if batch_texts:
        embs = embed_texts(batch_texts)
        for emb, (u, chunk_index, did, ctext) in zip(embs, batch_meta):
            batch_docs.append({
                "id": did,
                "entity_type": u["entity_type"],
                "title": u["title"],
                "content": ctext,

                "source_table": u["source_table"],
                "source_id": u["source_id"],
                "chunk_index": chunk_index,

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
