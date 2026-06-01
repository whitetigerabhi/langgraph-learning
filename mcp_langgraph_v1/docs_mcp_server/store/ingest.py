import json
from pathlib import Path


def load_docs():
    docs_path = Path(__file__).parent / "docs.json"
    with open(docs_path, "r", encoding="utf-8") as f:
        return json.load(f)