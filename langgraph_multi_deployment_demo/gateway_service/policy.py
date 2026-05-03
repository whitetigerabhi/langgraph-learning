import re

DESTRUCTIVE_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\btruncate\s+table\b",
    r"\bdelete\s+table\b",
    r"\bdestroy\s+repo\b",
    r"\brm\s+-rf\b",
    r"\bwipe\b",
]

def _matches_any(patterns, text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)

def preflight_rbac(role: str, query: str) -> tuple[bool, str]:
    role = (role or "user").lower()
    if _matches_any(DESTRUCTIVE_PATTERNS, query) and role != "admin":
        return (False, "BLOCK: Destructive actions are restricted to admins. Please try again with a non-destructive request.")
    return (True, "")