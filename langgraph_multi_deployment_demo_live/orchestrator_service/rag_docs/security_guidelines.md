# Security Guidelines (Internal)

## Secrets & Credentials
- Never paste secrets into chat, tickets, or logs.
- Use environment variables or a secrets manager.
- Rotate exposed keys immediately.

## Data Handling
- Treat all production identifiers as sensitive.
- Do not store PHI/PII in local files unless explicitly approved.

## Access Control
- Use role-based access controls (RBAC) for destructive actions.
- Prefer allow-lists over block-lists for high-risk operations.

## Logging
- Log request IDs, timestamps, and high-level outcomes.
- Do not log raw user content if it may contain sensitive data.