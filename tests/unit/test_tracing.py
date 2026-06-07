"""
Unit tests for the OpenTelemetry tracing utility.
"""

from __future__ import annotations

from unittest.mock import patch

from hybrid_rag.utils.tracing import (
    DummySpan,
    DummyTracer,
    get_tracer,
    initialize_tracing,
    start_span,
    trace_span,
)


class TestTracing:
    def test_dummy_tracer_and_span(self):
        tracer = DummyTracer()
        with tracer.start_as_current_span("test") as span:
            assert isinstance(span, DummySpan)
            span.set_attribute("key", "value")
            span.set_attributes({"a": 1})
            span.record_exception(Exception("err"))

    def test_get_tracer_fallback(self, monkeypatch):
        monkeypatch.setenv("TRACING_ENABLED", "false")
        tracer = get_tracer()
        assert isinstance(tracer, DummyTracer)

    def test_start_span_context_manager(self, monkeypatch):
        monkeypatch.setenv("TRACING_ENABLED", "false")
        with start_span("test_span", {"custom_attr": "hello"}) as span:
            assert isinstance(span, DummySpan)

    def test_trace_span_decorator(self, monkeypatch):
        monkeypatch.setenv("TRACING_ENABLED", "false")

        @trace_span("decorated_func", {"attr": "decorated"})
        def sample_function(x, y):
            return x + y

        assert sample_function(3, 4) == 7

    def test_initialize_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("TRACING_ENABLED", "false")
        initialize_tracing()
        # Should set dummy
        assert isinstance(get_tracer(), DummyTracer)

    def test_initialize_tracing_enabled(self, monkeypatch):
        monkeypatch.setenv("TRACING_ENABLED", "true")
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")

        with patch("opentelemetry.sdk.trace.TracerProvider.add_span_processor") as mock_add:
            initialize_tracing()
            assert mock_add.called

            tracer = get_tracer()
            assert not isinstance(tracer, DummyTracer)
