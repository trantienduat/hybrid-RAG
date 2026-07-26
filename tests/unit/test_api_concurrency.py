"""Concurrency tests for blocking retrieval behind the async API."""

from __future__ import annotations

import asyncio
from threading import Event, Timer
from unittest.mock import MagicMock

import pytest

from hybrid_rag.api.main import _retrieve_context
from hybrid_rag.api.schemas import QueryRequest
from hybrid_rag.retrieval.context_assembler import RetrievalContext


@pytest.mark.asyncio
async def test_retrieval_does_not_block_the_event_loop():
    retrieval_started = Event()
    release_retrieval = Event()
    retriever = MagicMock()

    def retrieve_with_context(*_args, **_kwargs):
        retrieval_started.set()
        assert release_retrieval.wait(timeout=1)
        return RetrievalContext(text="")

    retriever.retrieve_with_context.side_effect = retrieve_with_context
    request = QueryRequest(question="How does retrieval work?")
    timer = Timer(0.05, release_retrieval.set)
    timer.start()

    event_loop_turns = 0
    retrieval_finished = False

    async def count_event_loop_turns():
        nonlocal event_loop_turns
        while not retrieval_finished:
            event_loop_turns += 1
            await asyncio.sleep(0)

    counter = asyncio.create_task(count_event_loop_turns())
    try:
        await _retrieve_context(retriever, request)
        retrieval_finished = True
        await counter
    finally:
        retrieval_finished = True
        release_retrieval.set()
        timer.cancel()

    assert retrieval_started.is_set()
    assert event_loop_turns > 10
