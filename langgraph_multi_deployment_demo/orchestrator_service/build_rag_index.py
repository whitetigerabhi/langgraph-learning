import os
import json
import glob
import hashlib
from typing import List, Dict, Any
from openai import AzureOpenAI

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "rag_docs")
OUT_PATH = os.path.join(BASE_DIR, "storage", "rag_index.json")

CHUNK_SIZE = 900       # characters (prototype)
CHUNK_OVERLAP = 150    # characters (prototype)

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

def _embedding_deployment() -> str:
    dep = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "").strip()
    if not dep:
        raise RuntimeError(
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is not set. "
            "Create an embeddings deployment (e.g., text-embedding-3-small) and export it."
        )
    return dep

def _chunk_text(text: str) -> Listtext = (text or "").replace("\r\n", "\n")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + CHUNK_SIZE)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks

def _stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _embed_texts(texts: List[str]) -> List[List[float]]:
    client = _client()
    model = _embedding_deployment()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not files:
        raise SystemExit(f"No markdown files found in {DOCS_DIR}. Create rag_docs/*.md first.")

    records: List[Dict[str, Any]] = []
    batch_size = 16

    for path in files:
        doc_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = _chunk_text(content)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embs = _embed_texts(batch)

            for j, (txt, emb) in enumerate(zip(batch, embs)):
                chunk_index = i + j
                chunk_id = _stable_id(f"{doc_name}:{chunk_index}")
                records.append({
                    "doc": doc_name,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "text": txt,
                    "embedding": emb,
                })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f)

    print(f"✅ RAG index built: {len(records)} chunks → {OUT_PATH}")

if __name__ == "__main__":
    main()