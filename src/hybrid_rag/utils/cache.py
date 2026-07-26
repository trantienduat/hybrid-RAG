"""
Redis caching adapter — implements cache support for LLM/RAG responses.
Supports exact matching of queries and fails open gracefully if Redis is offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisQueryCache:
    """Async Redis cache client with fail-open fallback capability."""

    def __init__(self) -> None:
        self.host = os.environ.get("REDIS_HOST", "localhost")
        self.port = int(os.environ.get("REDIS_PORT", 6379))
        self.db = int(os.environ.get("REDIS_DB", 1))
        self.password = os.environ.get("REDIS_PASSWORD", None)
        self.socket_timeout = float(os.environ.get("REDIS_TIMEOUT", 2.0))

        self.client: aioredis.Redis | None = None
        self.is_connected = False

    async def connect(self) -> None:
        """Initialize connection pool and ping the server."""
        try:
            self.client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.socket_timeout,
                decode_responses=True,
            )
            # Perform active ping test
            await self.client.ping()
            self.is_connected = True
            logger.info("Connected to Redis cache at %s:%d (db=%d)", self.host, self.port, self.db)
        except Exception as exc:
            self.is_connected = False
            self.client = None
            logger.warning(
                "Redis cache connection failed. Operating in FAIL-OPEN mode (caching disabled): %s",
                exc,
            )

    async def disconnect(self) -> None:
        """Close connection pool cleanly."""
        if self.client:
            try:
                await self.client.aclose()
            except Exception:
                pass
            self.client = None
        self.is_connected = False

    async def get(self, key_str: str) -> Any | None:
        """Get deserialized JSON value for the given key. Fails open on errors."""
        if not self.is_connected or not self.client:
            return None
        try:
            val = await self.client.get(key_str)
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.warning("Redis GET request failed (failing open): %s", exc)
        return None

    async def set(self, key_str: str, val: Any, ttl_seconds: int = 3600) -> None:
        """Set serialized JSON value for key with TTL. Fails open on errors."""
        if not self.is_connected or not self.client:
            return
        try:
            val_str = json.dumps(val)
            await self.client.set(key_str, val_str, ex=ttl_seconds)
        except Exception as exc:
            logger.warning("Redis SET request failed (failing open): %s", exc)

    @staticmethod
    def generate_key(
        question: str,
        codebase_query: bool,
        repository: str | None,
        llm_model: str,
        top_k: int,
        context_n: int,
        max_tokens: int | None = None,
        max_chars: int | None = None,
        stream: bool = False,
    ) -> str:
        """Generate a stable, deterministic cache key from request parameters."""
        params_str = (
            f"v2|q:{question}|cq:{codebase_query}|repo:{repository or 'all'}|"
            f"model:{llm_model}|top:{top_k}|n:{context_n}|"
            f"max_tokens:{max_tokens}|max_chars:{max_chars}|stream:{stream}"
        )
        sha = hashlib.sha256(params_str.encode("utf-8")).hexdigest()
        return f"hybrid_rag:query_cache:{sha}"
