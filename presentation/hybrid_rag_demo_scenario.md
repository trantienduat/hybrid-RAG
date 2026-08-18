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
> Lệnh `scripts/reset_demo_checkout.py` sẽ tự động:
> 1. Xoá sạch toàn bộ nodes & edges của `demo-checkout` trong **FalkorDB**.
> 2. Xoá sạch toàn bộ vector chunks của `demo-checkout` trong **Qdrant**.
> 3. Xoá file mở rộng `logger.py` (nếu đã tạo trong lần demo trước).
> Giúp bạn reset hệ thống về trạng thái ban đầu chỉ trong **1 giây** để demo nhiều lần mà không sợ rác dữ liệu!

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
   > “Thưa Hội đồng, để đảm bảo tính khách quan và chứng minh hệ thống hoạt động hoàn toàn theo thời gian thực (không hardcode dữ liệu), chúng ta bắt đầu từ một trạng thái sạch.”

### Step 2: Trigger Live Indexing

Bạn có thể kích hoạt Indexing theo 1 trong 2 cách:

* **Cách A (Trực quan trên Web UI - Khuyên dùng):**
  1. Bấm nút **`+ New Project`** ở góc trên thanh Sidebar.
  2. Nhập:
     - **Repository Directory Path**: `fixtures/demo_checkout` (hoặc `/Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/fixtures/demo_checkout`)
     - **Namespace / Repository Name**: `demo-checkout`
  3. Bấm **Start Indexing**: Thanh tiến độ xuất hiện và đồ thị 3D tải ngay khi hoàn thành!

* **Cách B (Qua CLI Terminal):**
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
4. *Narration:*
   > “Chỉ trong vài giây, 2 file code thật đã được chuyển hoá đồng thời thành 2 góc nhìn: Đồ thị FalkorDB lưu trữ các liên kết gọi hàm chính xác, và Vector Qdrant lưu trữ ngữ nghĩa để tìm kiếm văn bản. Cả hai được gắn kết chặt chẽ bởi cùng một Provenance run_id.”

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
  > “Với câu hỏi về quan hệ gọi hàm, hệ thống đi thẳng vào Knowledge Graph. Quan hệ nào chứng minh được thì trả về kết quả; quan hệ nào chưa có mã nguồn như logger hay dynamic attribute thì hệ thống giữ nguyên dạng STUB chứ tuyệt đối không đoán mò.”

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
  > “Khi hỏi về luồng xử lý thanh toán, Semantic Search tìm ra file `payment.py` dù không có link cứng trực tiếp. Thuật toán RRF kết hợp bằng chứng từ cả 2 nguồn để LLM trả lời đầy đủ ngữ cảnh nghiệp vụ.”

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
  > “Với câu hỏi toàn cảnh, hệ thống không nhồi nhét mã nguồn của từng hàm vào context mà sử dụng bản tóm tắt kiến trúc theo thư mục (Directory-based community) được sinh sẵn.”

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
  > “Khi bổ sung file `logger.py`, tính năng Incremental Indexing chỉ phân tích diff của file mới trong vài chục mili-giây. Đồ thị 3D lập tức cập nhật node mới, và khi hỏi lại, hệ thống có ngay câu trả lời chính xác mà không cần re-index toàn bộ kho mã.”

---

## Act IV: Academic Boundaries & Closing Statement

To maintain academic rigor before the defense committee, state these boundaries explicitly:

1. **Teaching Fixture vs Benchmark:** `demo_checkout` is designed to visualize mechanisms (indexing, routing, stubs, incremental updates); the quantitative research claims (Answer Correctness +0.046, paired 95% range [0.002, 0.094]) are measured independently on the benchmark repository `llama-index-core`.
2. **Static Links vs Runtime Guarantee:** Graph edges reflect static AST facts, not runtime dynamic traces.
3. **Deterministic Partitioning:** Folder community grouping follows directory structure deterministically, avoiding non-deterministic Louvain clustering drift.

### Closing Sentence:
> 💬 *“Kịch bản demo trên vừa minh chứng trọn vẹn toàn bộ chu trình kỹ thuật của luận văn: từ phân tách mã nguồn thành 2 kho dữ liệu có nguồn gốc, định tuyến 3 đường tìm kiếm thông minh có nhận diện stub, cho đến khả năng cập nhật đồ thị tức thì qua incremental indexing.”*
