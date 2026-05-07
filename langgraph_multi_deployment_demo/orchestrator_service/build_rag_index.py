import os
import json
import glob
import hashlib
from typing import List, Dict, Any
from openai import AzureOpenAI

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "rag_docs")
OUT_PATH = os.path.join(BASE_DIR, "storage", "rag_index.json")

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150    # characters
BATCH_SIZE = 16


# -----------------------------
# CLIENT
# -----------------------------
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
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is not set.\n"
            "Run:\n"
            "export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=\"embedding-model\""
        )
    return dep


# -----------------------------
# TEXT CHUNKING
# -----------------------------
def _chunk_text(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n")

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

        # overlap logic
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


# -----------------------------
# UTIL
# -----------------------------
def _stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    client = _client()
    model = _embedding_deployment()

    resp = client.embeddings.create(
        model=model,
        input=texts
    )

    return [d.embedding for d in resp.data]


# -----------------------------
# MAIN BUILD
# -----------------------------
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))

    if not files:
        raise SystemExit(
            f"No markdown files found in {DOCS_DIR}. "
            "Create files in rag_docs/ first."
        )

    records: List[Dict[str, Any]] = []

    for path in files:
        doc_name = os.path.basename(path)

        print(f"📄 Processing: {doc_name}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = _chunk_text(content)

        print(f"   → {len(chunks)} chunks")

        # batch embedding
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embeddings = _embed_texts(batch)

            for j, (txt, emb) in enumerate(zip(batch, embeddings)):
                chunk_index = i + j

                records.append({
                    "doc": doc_name,
                    "chunk_index": chunk_index,
                    "chunk_id": _stable_id(f"{doc_name}:{chunk_index}"),
                    "text": txt,
                    "embedding": emb,
                })

    # save index
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f)

    print("\n✅ RAG index built successfully")
    print(f"📦 Total chunks: {len(records)}")
    print(f"📁 Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
