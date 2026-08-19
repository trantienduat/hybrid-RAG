# Hybrid-RAG Live Demo Scenario & Script

## Purpose & Overview

Walk the thesis defense committee through a complete, live, interactive demonstration of the Hybrid-RAG system in **5–7 minutes**. 

The demo proves the three core engineering and research mechanisms presented in the slides:
1. **Dual Representation & Shared Provenance (Core 05 / Slide 5.1):** From raw Python source to synchronized FalkorDB graph facts and Qdrant vector chunks.
2. **Dynamic Search Routing & Explicit Stubs (Core 06 / Slides 6.1–6.3):** How the Question Checker selects exact-link, local-behavior (RRF fusion), or whole-repository paths, while preserving honest uncertainty (`STUB`).
3. **Live Incremental Indexing & Atomic Updates (Core 08 / Slide 8.1):** How adding a new module triggers fast, diff-based incremental re-indexing without rebuilding the entire graph or invalidating safe caches.

---

## Fixture Structure

The live demo uses a small, transparent, and auditable fixture in `fixtures/demo_checkout/`:

### Base Fixture (2 files):
```text
fixtures/demo_checkout/
├── checkout.py    (Checkout class: calls _validate, _total, payment.charge, log.start/end)
└── payment.py     (PaymentGateway class: charge() calls _authorize, _capture)
```

### Extension Fixture (added during Act III):
```text
└── logger.py      (ConsoleLogger class: start(), end())
```

---

## Preflight & Setup

Run before the defense presentation to ensure all local containers and models are active and start from a clean, reproducible state:

```bash
cd /Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG

# 1. Reset Clean Slate (Reproducible Prep - Purge demo-checkout)
.venv/bin/python scripts/reset_demo_checkout.py

# 2. Start Docker containers (FalkorDB + Qdrant + API + MCP)
docker compose up -d

# 3. Verify local Ollama model
ollama list
```

> [!TIP]
> The `scripts/reset_demo_checkout.py` command automatically:
> 1. Purges all `demo-checkout` nodes and edges from **FalkorDB**.
> 2. Purges all `demo-checkout` vector chunks from **Qdrant**.
> 3. Deletes the extension file `logger.py` (if created in a previous demo run).
> This resets the environment to a pristine state in **~1 second**, enabling repeatable demonstrations without data remnants!

### Prepare Snippet File:
Create a ready-to-copy extension snippet `fixtures/demo_checkout/logger.py.snippet`:
```python
"""Logging service extension module for incremental index demo."""


class ConsoleLogger:
    def __init__(self, prefix: str = "[CHECKOUT]") -> None:
        self.prefix = prefix

    def start(self) -> None:
        print(f"{self.prefix} Transaction started")

    def end(self) -> None:
        print(f"{self.prefix} Transaction completed successfully")
```

---

## Act I: Clean Slate & Live Dual Indexing (Core 05)

**Objective:** Prove that the system does not use hardcoded or pre-baked databases, and visualize the real-time creation of dual representations.

### Step 1: Show Clean State
1. Open the Web Portal at `http://localhost:8000`.
2. Point to the repository selector: verify that `demo-checkout` does not exist yet.
3. *Narration:*
   > “Distinguished Committee members, to ensure objectivity and demonstrate that our system operates purely in real time without hardcoded or pre-baked data, we begin from a completely clean state.”

### Step 2: Trigger Live Indexing

You can trigger indexing via either method:

* **Option A (Interactive Web UI — Recommended):**
  1. Click the **`+ New Project`** button in the sidebar.
  2. Enter:
     - **Repository Directory Path**: `fixtures/demo_checkout` (or `/Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/fixtures/demo_checkout`)
     - **Namespace / Repository Name**: `demo-checkout`
  3. Click **Start Indexing**: The live progress bar appears and the interactive 3D graph loads immediately upon completion!

* **Option B (CLI Terminal):**
  ```bash
  .venv/bin/hybrid-rag index fixtures/demo_checkout \
    --repo-name demo-checkout \
    --rebuild \
    --no-incremental
  ```

Observe the real-time logs:
- Tree-sitter AST parsing $\rightarrow$ Extracting `Module`, `Class`, `Function` nodes.
- Entity Resolver $\rightarrow$ Classifying internal calls vs unresolved stubs.
- Dual emission $\rightarrow$ FalkorDB graph facts + Qdrant 768-d vector chunks with shared provenance.

### Step 3: Inspect 3D Graph & Dual Store Records
1. Switch to the Web UI: the interactive **3D Force Graph** renders immediately.
2. Rotate and zoom to show:
   - `checkout.py` $\rightarrow$ `Checkout` $\rightarrow$ `checkout()`
   - Internal edges: `CALLS → _validate`, `CALLS → _total`
   - `payment.py` $\rightarrow$ `PaymentGateway` $\rightarrow$ `charge()`
3. Open Qdrant Dashboard (`http://localhost:6333/dashboard`) or UI Chunks panel to show:
   - Vector chunk payload containing `node_id`, `span: L12-L17`, `file_path: checkout.py`, and `schema_version: 3`.

#### 🛠️ Or run these 2 Terminal commands to inspect data directly from both databases:

* **Command 1: Extract Knowledge Graph from FalkorDB (Nodes, Edges, Metadata)**
  ```bash
  .venv/bin/python -c "
  from hybrid_rag.graph.falkordb_store import FalkorDBStore
  f = FalkorDBStore()
  print('=== 1. REPOSITORY METADATA ===')
  print(f.get_repository_metadata('demo-checkout'))

  print('\n=== 2. GRAPH NODES ===')
  nodes = f.query(\"MATCH (n) WHERE n.repository = 'demo-checkout' OR n.repo = 'demo-checkout' RETURN labels(n)[0], n.id, n.file_path\").result_set or []
  for r in nodes:
      print(f'{r[0]:<18} | {r[1]} ({r[2]})')

  print('\n=== 3. GRAPH EDGES ===')
  edges = f.query(\"MATCH (a)-[r]->(b) WHERE (a.repository = 'demo-checkout' OR a.repo = 'demo-checkout') AND (b.repository = 'demo-checkout' OR b.repo = 'demo-checkout') RETURN a.id, type(r), b.id\").result_set or []
  for r in edges:
      print(f'{r[0]} --[{r[1]}]--> {r[2]}')
  "
  ```

* **Command 2: Extract Vector Chunks from Qdrant (Embeddings & Payloads)**
  ```bash
  .venv/bin/python -c "
  from qdrant_client.models import Filter, FieldCondition, MatchValue
  from hybrid_rag.vector.qdrant_store import QdrantStore
  q = QdrantStore()
  f = Filter(must=[FieldCondition(key='repository', match=MatchValue(value='demo-checkout'))])
  cnt = q._client.count(collection_name=q._collection, count_filter=f, exact=True).count
  print(f'=== QDRANT VECTORS ({cnt} points) ===')
  points, _ = q._client.scroll(collection_name=q._collection, scroll_filter=f, limit=50, with_payload=True)
  for p in points:
      lbl = p.payload.get('label', '')
      nid = p.payload.get('node_id', '')
      fp = p.payload.get('file_path', '')
      txt = p.payload.get('text', '').replace('\n', ' ')[:70]
      print(f'[{lbl:<8}] {nid:<45} | file: {fp:<12} | text: {txt}...')
  "
  ```

4. *Narration:*
   > “In just a few seconds, two raw source files were simultaneously transformed into dual complementary representations: the FalkorDB graph captures deterministic function call links, while Qdrant stores semantic vector embeddings for text retrieval. Both representations remain bound by the exact same provenance run_id.”

---

## Act II: Multi-Path Search & Honest Uncertainty (Core 06)

**Objective:** Demonstrate that different questions select distinct evidence paths, and observe the system's explicit handling of stubs without hallucination.

### Query 1: Exact-Link Path (Focus: Deterministic AST & Stubs)
* **Prompt:**
  ```text
  What does Checkout.checkout call?
  ```
* **System Execution:**
  - Question Checker detects relation word `call` and symbol `Checkout.checkout` $\rightarrow$ routes to **FalkorDB Cypher query**.
* **Audience Verification:**
  - Returns `_validate` and `_total` as confirmed graph edges.
  - Explicitly lists `self.payment.charge` and `self.log` as **`STUB`** (`__call__charge`, `__call__start`).
* **Narration:**
  > “For structural call-graph questions, the system routes directly into the Knowledge Graph. Confirmed relationships are returned with certainty; missing or external dependencies such as loggers or dynamic attributes are explicitly preserved as STUB references rather than hallucinated.”

---

### Query 2: Local-Behavior Path (Focus: RRF Fusion & Cross-File Context)
* **Prompt:**
  ```text
  How does checkout process payment?
  ```
* **System Execution:**
  - Question Checker identifies behavioral intent $\rightarrow$ queries both FalkorDB (graph links) and Qdrant (semantic chunks).
  - Retrieves `payment.py` chunk via vector search (semantic similarity on 'payment / charge').
  - Applies **RRF (Reciprocal Rank Fusion)** to rank, deduplicate, and limit context window.
* **Audience Verification:**
  - Shows combined context from `checkout.py` and `payment.py` passed to the LLM prompt.
  - Returns step-by-step business flow: `_validate → _total → charge (_authorize + _capture)`.
* **Narration:**
  > “When inquiring about payment processing behavior, Semantic Vector Search identifies `payment.py` even without an explicit direct call link. The RRF algorithm fuses evidence from both representations so the LLM receives complete, grounded domain context.”

---

### Query 3: Whole-Repository Path (Focus: Folder Community Summaries)
* **Prompt:**
  ```text
  How do checkout and payment work together in this repository?
  ```
* **System Execution:**
  - Question Checker detects repository-wide architectural query $\rightarrow$ reads pre-generated **Community Summary** from FalkorDB.
* **Audience Verification:**
  - Context contains high-level folder summary written by local Ollama model instead of polluting context with dozens of raw function bodies.
* **Narration:**
  > “For whole-repository architectural questions, the system avoids flooding the context window with dozens of individual function bodies, providing instead a pre-synthesized directory-based community summary.”

---

## Act III: Live Code Extension & Incremental Indexing (Core 08)

**Objective:** Prove live incremental indexing capability (Slide 08)—adding code updates the graph in milliseconds without full rebuilds.

### Step 1: Add Extension Code
Copy the snippet into the live fixture directory:
```bash
cp fixtures/demo_checkout/logger.py.snippet fixtures/demo_checkout/logger.py
```

### Step 2: Trigger Incremental Indexing
Trigger an incremental update via Web UI or CLI:
```bash
.venv/bin/hybrid-rag index fixtures/demo_checkout \
  --repo-name demo-checkout \
  --incremental
```

### Step 3: Verify the Incremental Update
1. **Performance Check:** Notice the operation completes in **< 100ms** (only the changed file `logger.py` is hashed and parsed; `checkout.py` and `payment.py` are skipped).
2. **Graph Visualizer Check:**
   - The 3D graph dynamically adds the new cluster: `logger.py` $\rightarrow$ `ConsoleLogger` $\rightarrow$ `start()`, `end()`.
3. **Re-query Test:**
   ```text
   How does Checkout log its operations?
   ```
   - The system immediately returns the answer grounded in the newly added `ConsoleLogger` methods.

* **Narration:**
  > “When adding `logger.py`, the incremental indexer only processes the single file diff in tens of milliseconds. The 3D graph dynamically updates with the new node, and re-querying immediately provides accurate answers without rebuilding the entire codebase index.”

---

## Act IV: Academic Boundaries & Closing Statement

To maintain academic rigor before the defense committee, state these boundaries explicitly:

1. **Teaching Fixture vs Benchmark:** `demo_checkout` is designed to visualize mechanisms (indexing, routing, stubs, incremental updates); the quantitative research claims (Answer Correctness +0.046, paired 95% range [0.002, 0.094]) are measured independently on the benchmark repository `llama-index-core`.
2. **Static Links vs Runtime Guarantee:** Graph edges reflect static AST facts, not runtime dynamic traces.
3. **Deterministic Partitioning:** Folder community grouping follows directory structure deterministically, avoiding non-deterministic Louvain clustering drift.

### Closing Sentence:
> 💬 *“This demonstration verifies the complete end-to-end technical pipeline of our thesis: from dual-store code decomposition with shared provenance and three-way intelligent evidence routing with stub awareness, to instantaneous graph updates powered by incremental indexing.”*
