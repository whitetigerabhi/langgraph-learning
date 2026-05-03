import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions

# Default: block Medium+ (often represented as >= 4 in the service severity scale)
CS_BLOCK_THRESHOLD = int(os.environ.get("CS_BLOCK_THRESHOLD", "4"))

def _client() -> ContentSafetyClient:
    endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
    key = os.environ["CONTENT_SAFETY_KEY"]
    return ContentSafetyClient(endpoint, AzureKeyCredential(key))

def analyze_text(text: str) -> dict:
    client = _client()
    resp = client.analyze_text(AnalyzeTextOptions(text=text))

    categories = {}
    max_sev = 0
    for item in resp.categories_analysis:
        sev = int(item.severity)
        categories[str(item.category)] = sev
        max_sev = max(max_sev, sev)

    return {"max_severity": max_sev, "categories": categories}

def should_block(text: str) -> tuple[bool, dict]:
    details = analyze_text(text)
    return (details["max_severity"] >= CS_BLOCK_THRESHOLD, details)