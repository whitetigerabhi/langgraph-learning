import os
import glob
import uuid
import hashlib
from typing import List
from openai import AzureOpenAI

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# ---------------- Config ----------------
DOCS_GLOB = "rag_docs/*.md"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))        # chars
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))  # chars
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]  # e.g. embedding-model

# ---------------- Clients ----------------
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

# ---------------- Helpers ----------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = (text or "").replace("\r\n", "\n")
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

def stable_chunk_id(doc_name: str, chunk_index: int) -> str:
    s = f"{doc_name}:{chunk_index}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]

# ---------------- Main ----------------
def main():
    paths = sorted(glob.glob(DOCS_GLOB))
    if not paths:
        raise SystemExit(f"No docs found at {DOCS_GLOB}. Create rag_docs/*.md first.")

    docs_to_upload = []

    for path in paths:
        doc_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"📄 {doc_name}: {len(chunks)} chunks")

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            embs = embed_texts(batch)

            for j, (chunk, emb) in enumerate(zip(batch, embs)):
                chunk_index = i + j
                docs_to_upload.append({
                    "id": str(uuid.uuid4()),
                    "doc": doc_name,
                    "chunk_index": chunk_index,
                    "chunk_id": stable_chunk_id(doc_name, chunk_index),
                    "content": chunk,
                    "embedding": emb,
                })

    print(f"⬆️ Uploading {len(docs_to_upload)} chunk docs to index '{INDEX_NAME}' ...")
    result = search.upload_documents(documents=docs_to_upload)

    failed = [r for r in result if not r.succeeded]
    if failed:
        print(f"⚠️ Upload finished with failures: {len(failed)}")
        print(failed[:3])
    else:
        print("✅ Upload complete (all documents succeeded)")

if __name__ == "__main__":
    main()