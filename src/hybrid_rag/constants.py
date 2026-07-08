import os

# Centralized default configurations for LLM and Embedding models
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-coder:7b")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
