import json


def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return text