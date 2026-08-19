from __future__ import annotations

MIN_CONTEXT_MATCH_LENGTH = 10
MIN_RELEVANCE_DOC_LENGTH = 10
MAX_DOC_PREVIEW_LENGTH = 2000
P95_PERCENTILE = 0.95

# Credential values that must never be accepted outside an explicit
# local/dev/test environment. Defined here (a dependency-free leaf) so both
# config validation and API auth share a single source of truth.
INSECURE_DEFAULT_KEYS = frozenset({"", "change-me", "sk-xxx", "your-api-key", "nano-rag-local"})
