import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Constants/Defaults
DEFAULT_FALKORDB_HOST = "localhost"
DEFAULT_FALKORDB_PORT = 6379
DEFAULT_FALKORDB_GRAPH = "codebase"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_QDRANT_COLLECTION = "code_chunks"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "gemma4:12b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_RRF_K = 60
DEFAULT_RRF_STRUCTURAL_W = 3.0
DEFAULT_RRF_HYBRID_W = 1.5
_LOCAL_OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def validate_local_ollama_url(url: str) -> str:
    """Require inference traffic to stay on this host (or its container bridge)."""
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in _LOCAL_OLLAMA_HOSTS:
        raise ValueError(
            "OLLAMA_BASE_URL must use http with localhost, 127.0.0.1, ::1, or host.docker.internal"
        )
    return url.rstrip("/")


def validate_local_provider_configuration() -> None:
    """Reject obsolete provider-selection flags instead of silently ignoring them."""
    configured = [name for name in ("LLM_PROVIDER", "EMBED_PROVIDER") if os.environ.get(name)]
    if configured:
        raise ValueError(
            f"Provider selection is no longer supported; remove {', '.join(configured)}. "
            "Hybrid-RAG uses local Ollama only."
        )


class Config:
    """Centralized configuration manager supporting config.json and env overrides."""

    def __init__(self, config_path: str | None = None) -> None:
        validate_local_provider_configuration()
        self.config_data: dict[str, Any] = {}
        self.config_file_path = self._resolve_config_path(config_path)
        self.load_config()

    def _resolve_config_path(self, override_path: str | None = None) -> Path | None:
        """Resolve the path to config.json, checking env vars and defaults."""
        if override_path:
            return Path(override_path)

        env_path = os.environ.get("HYBRID_RAG_CONFIG")
        if env_path:
            return Path(env_path)

        # Check standard locations: current directory, or parent directory
        cwd_path = Path("config.json")
        if cwd_path.exists():
            return cwd_path

        alt_path = Path("hybrid_rag_config.json")
        if alt_path.exists():
            return alt_path

        # Check in project root if running from a subdirectory
        try:
            root_path = Path(__file__).resolve().parents[2] / "config.json"
            if root_path.exists():
                return root_path
        except Exception:
            pass

        return None

    def load_config(self) -> None:
        """Load and parse config file if it exists."""
        if not self.config_file_path or not self.config_file_path.exists():
            logger.info("No configuration file found. Falling back to environment variables.")
            self.config_data = {}
            return

        try:
            with open(self.config_file_path, encoding="utf-8") as f:
                self.config_data = json.load(f)
            logger.info(f"Loaded configuration from {self.config_file_path}")
        except Exception as exc:
            logger.warning(
                f"Failed to parse config file at {self.config_file_path}: {exc}. Fail-open: continuing."
            )
            self.config_data = {}

    def get_val(self, key_path: str, default: Any = None) -> Any:
        """Get config value from config_data dict by dot path (e.g. 'falkordb.host')."""
        parts = key_path.split(".")
        curr = self.config_data
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return default
        return curr

    # Resolved settings (Env var > config.json > Default)
    @property
    def falkordb_host(self) -> str:
        return (
            os.environ.get("FALKORDB_HOST")
            or self.get_val("falkordb.host")
            or DEFAULT_FALKORDB_HOST
        )

    @property
    def falkordb_port(self) -> int:
        val = os.environ.get("FALKORDB_PORT") or self.get_val("falkordb.port")
        try:
            return int(val) if val else DEFAULT_FALKORDB_PORT
        except ValueError:
            return DEFAULT_FALKORDB_PORT

    @property
    def falkordb_graph(self) -> str:
        return (
            os.environ.get("FALKORDB_GRAPH")
            or self.get_val("falkordb.graph")
            or DEFAULT_FALKORDB_GRAPH
        )

    @property
    def qdrant_host(self) -> str:
        return os.environ.get("QDRANT_HOST") or self.get_val("qdrant.host") or DEFAULT_QDRANT_HOST

    @property
    def qdrant_port(self) -> int:
        val = os.environ.get("QDRANT_PORT") or self.get_val("qdrant.port")
        try:
            return int(val) if val else DEFAULT_QDRANT_PORT
        except ValueError:
            return DEFAULT_QDRANT_PORT

    @property
    def qdrant_collection(self) -> str:
        return (
            os.environ.get("QDRANT_COLLECTION")
            or self.get_val("qdrant.collection")
            or DEFAULT_QDRANT_COLLECTION
        )

    @property
    def ollama_url(self) -> str:
        url = (
            os.environ.get("OLLAMA_BASE_URL")
            or self.get_val("ollama.base_url")
            or DEFAULT_OLLAMA_URL
        )
        return validate_local_ollama_url(url)

    @property
    def llm_model(self) -> str:
        return os.environ.get("LLM_MODEL") or self.get_val("llm.model") or DEFAULT_LLM_MODEL

    @property
    def embed_model(self) -> str:
        return (
            os.environ.get("EMBED_MODEL") or self.get_val("embedding.model") or DEFAULT_EMBED_MODEL
        )

    @property
    def rrf_k(self) -> int:
        val = os.environ.get("RRF_K") or self.get_val("rrf.k")
        try:
            return int(val) if val else DEFAULT_RRF_K
        except ValueError:
            return DEFAULT_RRF_K

    @property
    def rrf_structural_weight(self) -> float:
        val = os.environ.get("RRF_STRUCTURAL_WEIGHT") or self.get_val("rrf.structural_weight")
        try:
            return float(val) if val else DEFAULT_RRF_STRUCTURAL_W
        except ValueError:
            return DEFAULT_RRF_STRUCTURAL_W

    @property
    def rrf_hybrid_weight(self) -> float:
        val = os.environ.get("RRF_HYBRID_WEIGHT") or self.get_val("rrf.hybrid_weight")
        try:
            return float(val) if val else DEFAULT_RRF_HYBRID_W
        except ValueError:
            return DEFAULT_RRF_HYBRID_W

    @property
    def repositories(self) -> list[dict[str, str]]:
        """Get defined repositories list from config.json."""
        repos = self.get_val("repositories", [])
        if not isinstance(repos, list):
            logger.warning(
                "Config value 'repositories' is not a list. Fail-open: using empty list."
            )
            return []

        parsed_repos = []
        for r in repos:
            if isinstance(r, dict) and "name" in r and "path" in r:
                parsed_repos.append({"name": str(r["name"]), "path": str(r["path"])})
        return parsed_repos

    def get_repo_path(self, repo_name: str) -> str | None:
        """Find local path for a repository by its name in config."""
        for r in self.repositories:
            if r["name"] == repo_name:
                return r["path"]
        return None


# Global configuration instance
app_config = Config()


def translate_path_for_docker(path: str | None) -> str | None:
    """Translate host-level paths to Docker volume-mounted paths if inside container."""
    if not path:
        return path
    if not os.path.exists("/Volumes/Kioxia_SSD") and os.path.isdir("/codebases"):
        if path.startswith("/Volumes/Kioxia_SSD/SSD_workspace/Personal"):
            return path.replace("/Volumes/Kioxia_SSD/SSD_workspace/Personal", "/codebases")
    return path
