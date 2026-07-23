"""
LLM Observability and Tracing Utility based on OpenTelemetry.
Integrates with Arize Phoenix, Langfuse, or any OTLP-compatible collector.
Supports graceful fallback if opentelemetry dependencies are not installed.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

# OpenTelemetry global handles
_tracer: Any = None
_provider: Any = None


def _is_tracing_enabled() -> bool:
    return os.environ.get("TRACING_ENABLED", "false").lower() in ("true", "1")


def _should_launch_server() -> bool:
    return os.environ.get("PHOENIX_LAUNCH_SERVER", "false").lower() in ("true", "1")


def _get_otlp_endpoint() -> str:
    return (
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or "http://localhost:4317"
    )


CallableType = TypeVar("CallableType", bound=Callable[..., Any])


class DummySpan:
    """Fallback span that does nothing when tracing is disabled."""

    def __enter__(self) -> DummySpan:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


class DummyTracer:
    """Fallback tracer that returns dummy spans."""

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[DummySpan, None, None]:
        yield DummySpan()


def initialize_tracing() -> None:
    """
    Initialize OpenTelemetry SDK and point it to the configured OTLP collector.
    Optionally launches the Arize Phoenix app server programmatically.
    """
    global _tracer, _provider

    if not _is_tracing_enabled():
        logger.debug("Tracing is disabled. Using dummy tracer.")
        _tracer = DummyTracer()
        return

    # Programmatic Arize Phoenix app launching if requested
    if _should_launch_server():
        try:
            import phoenix as px

            px.launch_app()
            logger.info("Arize Phoenix application server launched programmatically!")
        except Exception as exc:
            logger.warning("Failed to launch Arize Phoenix server inline: %s", exc)

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(attributes={"service.name": "hybrid-rag"})
        _provider = TracerProvider(resource=resource)

        otlp_endpoint = _get_otlp_endpoint()

        # Decide OTLP Exporter: gRPC or HTTP
        if "4317" in otlp_endpoint or not otlp_endpoint.startswith("http"):
            # Use gRPC
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                logger.info("Configured OTLP gRPC trace exporter to %s", otlp_endpoint)
            except ImportError:
                # Fallback to HTTP
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                logger.info("Configured OTLP HTTP trace exporter to %s", otlp_endpoint)
        else:
            # Use HTTP
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            logger.info("Configured OTLP HTTP trace exporter to %s", otlp_endpoint)

        _provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_provider)
        _tracer = trace.get_tracer("hybrid-rag")
        logger.info("OpenTelemetry tracing initialized successfully.")

        # Initialize OTel Logging Provider
        try:
            from opentelemetry._logs import set_logger_provider
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

            # Import the correct log exporter based on endpoint type
            if "4317" in otlp_endpoint or not otlp_endpoint.startswith("http"):
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            else:
                from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

            log_provider = LoggerProvider(resource=resource)
            set_logger_provider(log_provider)

            log_exporter = OTLPLogExporter(endpoint=otlp_endpoint)
            log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

            # Add OTel log handler to Python root logger
            handler = LoggingHandler(level=logging.INFO, logger_provider=log_provider)
            logging.getLogger().addHandler(handler)
            logger.info("OpenTelemetry logging handler initialized successfully.")
        except Exception as log_exc:
            logger.warning("Failed to initialize OpenTelemetry logging SDK: %s", log_exc)

        # Configure console formatter to print trace_id and span_id
        try:

            class OTelConsoleFormatter(logging.Formatter):
                def format(self, record):
                    span_ctx = trace.get_current_span().get_span_context()
                    if span_ctx and span_ctx.is_valid:
                        record.trace_id = trace.format_trace_id(span_ctx.trace_id)
                        record.span_id = trace.format_span_id(span_ctx.span_id)
                    else:
                        record.trace_id = "0"
                        record.span_id = "0"
                    return super().format(record)

            fmt = OTelConsoleFormatter(
                "[%(asctime)s] %(levelname)s [%(name)s] [trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s"
            )
            # Apply to all StreamHandlers on the root logger and uvicorn loggers
            for log_name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
                for h in logging.getLogger(log_name).handlers:
                    if isinstance(h, logging.StreamHandler):
                        h.setFormatter(fmt)
            logger.info("Console trace log formatter configured successfully.")
        except Exception as fmt_exc:
            logger.warning("Failed to configure console trace log formatter: %s", fmt_exc)

    except Exception as exc:
        logger.warning(
            "Failed to initialize OpenTelemetry SDK: %s. Falling back to dummy tracer.", exc
        )
        _tracer = DummyTracer()


def get_tracer() -> Any:
    """Retrieve the active tracer instance (or DummyTracer)."""
    global _tracer
    if _tracer is None:
        initialize_tracing()
    return _tracer


@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """
    Context manager to execute code block inside a custom traced span.
    Automatically captures exceptions.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


def trace_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Callable[[CallableType], CallableType]:
    """
    Decorator to wrap any function execution inside a traced span.
    Injects custom span metadata.
    """

    def decorator(func: CallableType) -> CallableType:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_span(name, attributes) as span:
                # Try to record input parameters as attributes
                if attributes is None:
                    # Log some generic info
                    span.set_attribute("function.name", func.__name__)
                result = func(*args, **kwargs)
                return result

        return cast(CallableType, wrapper)

    return decorator
