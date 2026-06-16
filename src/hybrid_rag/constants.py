import os

# Centralized default configurations for LLM and Embedding models
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:12b")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
