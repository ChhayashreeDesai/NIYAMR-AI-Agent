"""NIYAMR AI agent package (copied into without_api for isolation).

This mirrors the top-level `agent` package so `without_api` is self-contained.
"""

# Keep the no-API package purely local/deterministic: expose only non-API modules
__all__ = ["extractor", "summarizer", "sections", "rule_checks", "utils", "summarizer_hybrid"]
