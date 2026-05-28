FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for compiling tree-sitter binary bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration files
COPY pyproject.toml .

# Copy core source files
COPY src/ ./src/

# Install the application and its API dependencies natively inside the container
RUN pip install --no-cache-dir .[api]

# Expose the FastAPI server default port
EXPOSE 8000

# Set default connection environment variables (overridden by docker-compose)
ENV FALKORDB_HOST=falkordb
ENV FALKORDB_PORT=6379
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Run the API server as the container startup command
CMD ["hybrid-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
