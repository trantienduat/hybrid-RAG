# Product Requirement Prompt (PRP) — LLM Observability & Cloud Integration

## 🎯 Role & Objective
You are an expert DevOps and cloud infrastructure engineer AI. Your task is to implement **LLM Observability and Cloud Integration** for the `hybrid-RAG` platform. You must build an OpenTelemetry tracing system, configure an Arize Phoenix dockerized server with data persistence, and write Google Gemini API adapters for embedding generation and relationship extraction.

---

## 🏛️ Tech Stack & Standards
*   **Standards:** OpenTelemetry (OTel), OpenInference (for LLM metadata conventions).
*   **Infrastructure:** Docker Compose, Dockerfile.
*   **APIs:** Google Gemini Cloud API (GenerateContent & EmbedContent endpoints).
*   **Libraries:** `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `arize-phoenix` (optional/local).

---

## 🛠️ Functional Requirements

### 1. OpenTelemetry Tracing Layer ([utils/tracing.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/utils/tracing.py))
*   Implement `initialize_tracing()`:
    *   Expose variables: `TRACING_ENABLED` (bool), `PHOENIX_COLLECTOR_ENDPOINT` (defaults to `http://localhost:6006/v1/traces`).
    *   Set up a `TracerProvider` with standard batch span processor using OTLPSpanExporter (supporting both HTTP and gRPC protocols).
*   Implement `start_span(name, attributes)` context manager and `@trace_span` decorators.
*   Implement a **Dummy fallback**: If OTel libraries are missing or tracing is disabled, return a `DummyTracer` yielding no-op spans to prevent application failures.

### 2. Gemini API Integration
*   **Gemini Embedder** ([ingestion/gemini_embedder.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/gemini_embedder.py)):
    *   Implement `BaseEmbedder` interface targeting Google's `text-embedding-004` (768-dim) model.
    *   Support `embed_texts` batch requests using the `batchEmbedContents` REST endpoint.
*   **Gemini Extractor** ([ingestion/gemini_llm_extractor.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/gemini_llm_extractor.py)):
    *   Implement `BaseLLMExtractor` interface targeting `gemini-1.5-flash` to extract relationship triplets.
*   **Common Safety & Telemetry:**
    *   Both adapters must raise `ValueError` on initialization if `GEMINI_API_KEY` is not configured.
    *   Both adapters must use exponential backoff retries catching specific HTTP status errors (retrying on 429 and 5xx only) and connection timeouts.
    *   Both adapters must attach latency and token count metrics (`prompt_tokens`, `candidates_tokens`, `total_tokens`) as OpenInference span attributes.

### 3. Standalone Arize Phoenix Service (`docker-compose.yml`)
*   Define a `phoenix` service pulling `arizephoenix/phoenix:latest` exposing ports `6006` and `4317`.
*   Configure a named docker volume `phoenix_data` mapping to `/root/.phoenix` inside the container to persist collected traces across restarts.
*   Inject `TRACING_ENABLED=true` and `PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces` to application container environments.

### 4. Dependency Updates (`Dockerfile` & CI/CD)
*   Define `observability` optional dependency group in `pyproject.toml`.
*   Configure `Dockerfile` and release workflow `.github/workflows/release.yml` to compile and build using `.[api,mcp,observability]` to ensure telemetry dependencies are bundled at build-time.

---

## 📈 Verification
*   Implement unit tests verifying that enqueuing tracing-enabled processes runs mocked spans correctly.
