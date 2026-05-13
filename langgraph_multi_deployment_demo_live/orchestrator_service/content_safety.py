import os
from typing import Tuple, Dict, Any

CS_BLOCK_THRESHOLD = int(os.environ.get("CS_BLOCK_THRESHOLD", "4"))

_AZURE_SDK_OK = True
_AZURE_IMPORT_ERR = ""

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.contentsafety import ContentSafetyClient
    from azure.ai.contentsafety.models import AnalyzeTextOptions
except Exception as e:
    _AZURE_SDK_OK = False
    _AZURE_IMPORT_ERR = str(e)


def _client():
    if not _AZURE_SDK_OK:
        raise RuntimeError(
            "Azure Content Safety SDK is not available. Install: "
            "python3 -m pip install --user azure-ai-contentsafety azure-core. "
            f"Import error: {_AZURE_IMPORT_ERR}"
        )
    endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
    key = os.environ["CONTENT_SAFETY_KEY"]
    return ContentSafetyClient(endpoint, AzureKeyCredential(key))


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Azure AI Content Safety Analyze Text API: returns severity per harm category. [1](https://oneuptime.com/blog/post/2026-02-16-how-to-configure-content-filtering-policies-in-azure-openai-service/view)[2](https://developers.openai.com/api/reference/resources/moderations/methods/create)
    """
    client = _client()
    resp = client.analyze_text(AnalyzeTextOptions(text=text))

    categories: Dict[str, int] = {}
    max_sev = 0
    for item in resp.categories_analysis:
        sev = int(item.severity)
        categories[str(item.category)] = sev
        max_sev = max(max_sev, sev)

    return {"max_severity": max_sev, "categories": categories}


def should_block(text: str) -> Tuple[bool, Dict[str, Any]]:
    details = analyze_text(text)
    return (details["max_severity"] >= CS_BLOCK_THRESHOLD, details)
