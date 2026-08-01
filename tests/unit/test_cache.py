"""Unit tests for response-cache key isolation."""

from unittest.mock import AsyncMock

import pytest

from hybrid_rag.utils.cache import RedisQueryCache


def _key(**overrides):
    params = {
        "question": "How does retrieval work?",
        "codebase_query": True,
        "repository": "hybrid-rag",
        "llm_model": "test-model",
        "top_k": 20,
        "context_n": 5,
        "max_tokens": None,
        "max_chars": None,
        "stream": False,
    }
    params.update(overrides)
    return RedisQueryCache.generate_key(**params)


def test_cache_key_is_stable_for_identical_requests():
    assert _key() == _key()


def test_cache_key_separates_response_formats():
    assert _key(stream=False) != _key(stream=True)


def test_cache_key_separates_token_budgets():
    assert _key(max_tokens=512) != _key(max_tokens=2048)


def test_cache_key_separates_character_budgets():
    assert _key(max_chars=2000) != _key(max_chars=8000)


def test_cache_key_separates_index_generations():
    assert _key(cache_generation="4") != _key(cache_generation="5")


@pytest.mark.asyncio
async def test_bump_generation_atomically_invalidates_prior_keys():
    cache = RedisQueryCache()
    cache.client = AsyncMock()
    cache.is_connected = True

    assert await cache.bump_generation() is True

    cache.client.incr.assert_awaited_once_with("hybrid_rag:query_cache:generation")


@pytest.mark.asyncio
async def test_generation_failure_disables_cache_to_prevent_stale_reads():
    cache = RedisQueryCache()
    cache.client = AsyncMock()
    cache.client.incr.side_effect = RuntimeError("redis unavailable")
    cache.is_connected = True

    assert await cache.bump_generation() is False
    assert cache.is_connected is False
