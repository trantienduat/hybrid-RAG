# Hybrid-RAG demo scenario

## Purpose

Show one complete, understandable path from a local code snapshot to a grounded answer, then demonstrate that unsupported Java indexing is rejected explicitly.

Target duration: 7–10 minutes.

## What the audience should see

1. FalkorDB and Qdrant are healthy.
2. The question is scoped to one indexed repository.
3. The answer cites relevant code context rather than receiving the whole repository.
4. Graph relationships and vector similarity are complementary.
5. Java is rejected instead of being silently treated as supported Python.

## Preflight

Run this before the presentation. Do not start a fresh large indexing job immediately before the demo.

```bash
cd /Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG
docker compose up -d
ollama list
.venv/bin/hybrid-rag status
```

The prepared answer-quality index uses:

```bash
REPO=llama-core-answer-v3
```

The API uses the graph and Qdrant collection configured by the running service; the request supplies the repository scope.

If the long answer-quality benchmark is still using Ollama, prefer the recorded output below or wait for it to finish. A live demo query competes for the same local model capacity.

## Live path A — grounded code question

Run one question about a source flow the audience can follow:

```bash
QUESTION='What sequence does BaseRetriever.retrieve follow when it receives a plain query string, including callbacks and recursive retrieval?'
jq -n --arg question "$QUESTION" --arg repository "$REPO" \
  '{question:$question, repository:$repository, top_k:20, context_n:5, codebase_query:true, stream:false}' \
  | curl -sS http://127.0.0.1:8000/query \
    -H 'Content-Type: application/json' \
    -d @- \
  | jq '{answer, sources, query_type, latency_ms}'
```

Narrate the output in this order:

1. The query is repository-scoped.
2. Vector retrieval finds semantically related retriever code.
3. Graph retrieval contributes callback and recursive-retrieval relationships.
4. RRF combines the ranked candidates.
5. The API context assembler limits what reaches local Ollama.
6. The response contains the generated answer and source paths.

Do not claim that the answer is universally correct from one demo. Say that the independent answer-quality benchmark measures faithfulness, relevancy, and correctness across repeated cases.

## Live path B — fail-closed Java boundary

Create a temporary Java fixture without touching the repository:

```bash
DEMO_JAVA_DIR="$(mktemp -d /tmp/hybrid-rag-java-demo.XXXXXX)"
printf 'class Demo { void run() {} }\n' > "$DEMO_JAVA_DIR/Demo.java"
```

Attempt indexing:

```bash
.venv/bin/hybrid-rag index "$DEMO_JAVA_DIR" --repo-name demo-java
```

Expected behavior: the command rejects the project with an explicit unsupported-language error. Explain that a false-success index would be more dangerous than a clear rejection because it could produce misleading answers.

Clean up the temporary fixture after the demo:

```bash
rm -rf "$DEMO_JAVA_DIR"
```

## Optional evidence slide

Show the completed benchmark artifact, with its generated and scored counts:

```bash
jq -r '
  .records as $r |
  ([ $r[] | select(.faithfulness != null and .answer_relevancy != null and .answer_correctness != null) ] | length) as $s |
  "generated=" + (($r|length)|tostring) + ", scored=" + ($s|tostring)
' results/answer_quality_benchmark_schema3.json
```

Expected completed artifact: `generated=180, scored=180`. Use the checked-in statistics file for the confidence intervals; do not present a stale checkpoint as final evidence.

## Recovery plan

- If Ollama is slow: use a saved output and explain the same retrieval path.
- If the prepared index is unavailable: run `status` and present the architecture slide instead of indexing live.
- If the Java command does not reject: stop the demo and treat it as a defect; do not continue with a misleading result.
- If the benchmark is active: do not run another full evaluation concurrently.

## Closing sentence

“The important result is not only that the system can answer a code question; it can show which repository snapshot and retrieval context produced the answer, and it refuses unsupported input instead of pretending it succeeded.”
