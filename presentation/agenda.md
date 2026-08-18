# Thời lượng: 25 phút Thuyết trình + Demo; 15 phút Q&A

Đây là buổi bảo vệ Luận văn Thạc sĩ (Thesis Defense), không phải buổi ra mắt sản phẩm (Product Launch). Trọng tâm: Phương pháp luận, Thiết kế Thực nghiệm và Độ tin cậy Kết quả.

---
## ⏱️ THỜI GIAN BIỂU TỔNG QUAN (25 Phút)

[1. Problem, gap, related work, contributions]         ──>  4 Phút
[2. Methodology, architecture and retrieval mechanism]  ──>  8 Phút
[3. Experimental design and evaluation]                ──>  4 Phút
[4. Results, statistical interpretation, limitations]  ──>  5 Phút
[5. Focused live demo]                                 ──>  3 Phút
[6. Conclusion]                                        ──>  1 Phút
─────────────────────────────────────────────────────────────
[7. Q&A với Hội đồng / Giám khảo]                      ──> 15 Phút


---
## CHI TIẾT TỪNG PHẦN

### 1. BỐI CẢNH, RELATED WORK & ĐÓNG GÓP (4 Phút)
- **Problem Formulation & Research Gap**:
  - Tại sao Vector-only không đủ để trả lời các câu hỏi về quan hệ cấu trúc trong code (call, inherits)?
  - Khoảng trống nghiên cứu/kỹ thuật mà hệ thống Hybrid-RAG Local giải quyết.
- **Related Work (Literature Positioning)**:
  Định vị nghiên cứu trong bức tranh lớn hơn. Ghi rõ: đây là literature positioning, KHÔNG phải baseline đã chạy thực nghiệm.

  | Approach | Điểm mạnh | Giới hạn |
  |---|---|---|
  | BM25 | Lexical matching | Không hiểu semantic; không biểu diễn structural relation |
  | Dense/vector retrieval | Semantic similarity | Không biểu diễn quan hệ call/inheritance một cách explicit |
  | CodeBERT/GraphCodeBERT | Pretrained code representation | Không phải end-to-end repository retrieval system |
  | Microsoft GraphRAG | Graph/community-based corpus reasoning | Không được thiết kế quanh AST/code semantics |
  | **Hybrid-RAG (this work)** | Graph + semantic + provenance | Chi phí retrieval và độ phức tạp cao hơn |

- **Research Objectives & RQs**:
  - RQ1: Answer quality (Faithfulness, Relevancy, Correctness).
  - RQ2: Retrieval & Generation efficiency (Latency, Token footprint).
  - RQ3: Evidence validity and statistical confidence (95% CI).
- **Contribution Boundary (Phân loại thận trọng)**:
  - **Method/System Contribution**: Hệ thống truy xuất mã nguồn lai (Hybrid Code Retrieval) theo phạm vi repository có bảo toàn tính toàn vẹn (Provenance).
  - **Empirical Contribution**: Đánh giá thực nghiệm giữa Hybrid-RAG vs Vector-only bằng paired bootstrap benchmark.
  - **Engineering Capabilities** (không phải research contribution): FalkorDB, Qdrant, MCP Server, Web UI 2D/3D, Incremental Git Sync.

### 2. PHƯƠNG PHÁP LUẬN, KIẾN TRÚC & CƠ CHẾ TRUY XUẤT (8 Phút)
Tách biệt rõ 2 luồng xử lý cốt lõi:
- **Index-time Flow (Nạp dữ liệu)**:
  - Source snapshot & Phạm vi (Python-only).
  - AST parsing (Tree-sitter) & Trích xuất thực thể (Entity).
  - FQN (Fully Qualified Name) Entity resolution (Giải quyết xung đột namespace nội bộ).
  - Cấu trúc Graph: Cạnh `DEFINES`, `CALLS`, `IMPORTS`, `INHERITS`.
  - Vector chunking & Embedding (Qdrant - `nomic-embed-text`).
  - Đảm bảo tính toàn vẹn (Provenance: Commit hash, Index run ID, Schema version, Embedding model ID).
- **Query-time Flow (Định tuyến 3 luồng)**:
  - *Exact relation route*: Tìm kiếm quan hệ chính xác → Ưu tiên Graph traversal; fallback sang Vector nếu không tìm thấy target.
  - *Generic local route*: Truy vấn cục bộ thông thường → Kết hợp Graph + Vector + RRF.
  - *Global/community route*: Câu hỏi tổng quan kiến trúc → Truy xuất Community Summaries từ FalkorDB.
- **RRF (Reciprocal Rank Fusion)**: Công thức **hợp nhất thứ hạng (Rank)** từ hai luồng retrieval (Không dùng raw score vì thang đo Graph và Vector khác nhau):
  $$RRF\_Score(d) = \sum_{r \in \{graph, vector\}} \frac{W_r}{k + rank_r(d)}$$
- **Context Assembly**: Lắp ráp ngữ cảnh với token budget (Giới hạn token gửi cho LLM).
- **Local Generation**: Đẩy prompt vào Local LLM (`gemma4:12b`).

### 3. THIẾT KẾ THỰC NGHIỆM (4 Phút)
- **Experimental Design**:
  - Confirmatory workload: `llama-index-core==0.14.21` (30 câu hỏi: 10 simple, 10 medium, 10 hard).
  - Transformers & LangChain đã chạy **post-submission supplementary source-anchored slices** (10 case pairs mỗi repo); không nằm trong evidence hoặc confirmatory statistical conclusion của thesis đã nộp.
  - Source identity & Checksum (Đảm bảo minh bạch).
  - Baselines: Condition Vector-only vs. Hybrid-RAG.
  - Số lần chạy: 30 câu hỏi × 2 modes × 3 repeats = **180 raw records**.
  - *Đơn vị suy luận thống kê*: **30 cặp câu hỏi** (repeats averaged within case), không phải 180 câu trả lời độc lập.
  - Metrics: RAGAS (Faithfulness, Answer Relevancy, Correctness).
  - Judge: Automated judge (`qwen2.5-coder:7b`), không phải human adjudication.
  - Cùng answer model, prompt settings, top_k, context_n cho cả 2 conditions.
- **Method of Analysis**: Confirmatory LlamaIndex dùng paired bootstrap resampling (10,000 resamples, seed 42) trên 30 case pairs để tính 95% Confidence Intervals (CI). Supplementary Transformers/LangChain hiện báo cáo descriptive means/deltas trên 10 case pairs mỗi repo; không dùng cho pooled CI hoặc kết luận generalization.

### 4. KẾT QUẢ, DIỄN GIẢI THỐNG KÊ & HẠN CHẾ (5 Phút)
- **Kết quả (Results) — Trình bày ĐẦY ĐỦ cả 3 metrics**:

  | Metric | Vector mean | Hybrid mean | Δ Absolute | 95% CI | Kết luận |
  |---|---|---|---|---|---|
  | Faithfulness | 0.896 | 0.924 | +0.029 | [-0.069, 0.115] | **Chưa kết luận được** (CI chứa 0) |
  | Answer Relevancy | 0.729 | 0.808 | +0.079 | [-0.0001, 0.175] | **Borderline** (CI chạm 0) |
  | Answer Correctness | 0.652 | 0.698 | +0.046 | [0.002, 0.094] | **Có bằng chứng cải thiện** (CI loại trừ 0) |

- **RQ2 — Efficiency trade-off (confirmatory workload; case-level means)**:

  | Measure | Vector mean | Hybrid mean | Δ Hybrid − Vector | 95% CI của Δ |
  |---|---:|---:|---:|---:|
  | Retrieval latency | 134 ms | 371 ms | +237 ms | [+208, +266] |
  | Generation latency | 96.8 s | 87.3 s | −9.5 s | [−23.0 s, +1.4 s] |
  | Prompt tokens | 1,841 | 1,740 | −101 | [−282, +77] |
  | Completion tokens | 934 | 834 | −100 | [−259, +30] |

  - Retrieval trung bình 371 ms, dưới 1 giây trong workload đo được; đây không phải claim về end-to-end response SLA.

- **Diễn giải Thống kê (Statistical Interpretation)**:
  - Chỉ Answer Correctness có khoảng tin cậy hoàn toàn trên 0.
  - Với dữ liệu hiện tại, Faithfulness và Answer Relevancy chưa thiết lập được hiệu ứng khác 0; nghiên cứu lớn hơn trên nhiều câu hỏi và repository là future work.
  - Hybrid có retrieval overhead đo được; mức giảm generation/token là point estimate, nhưng CI hiện tại vẫn chứa 0.
- **Hạn chế (Threats to Validity)**:
  - Confirmatory benchmark trên 1 repo duy nhất (`llama-index-core`).
  - Transformers/LangChain là post-submission source-anchored slices, chỉ dùng làm supplementary replication; không thay thế confirmatory workload và không chứng minh generalization. Mở rộng lên 100 câu hỏi/full-repository vẫn là future work.
  - Chỉ hỗ trợ AST Python, chưa khái quát cho Java/Go.
  - Độ trễ generation LLM local cao (~87s) do giới hạn phần cứng suy luận.
  - Judge là automated (LLM), chưa có independent human annotation.
  - BM25, graph-only, CodeBERT, GraphRAG chưa được chạy thực nghiệm so sánh.

### 5. FOCUSED LIVE DEMO (3 Phút)
- **Mục tiêu**: Rút gọn tối đa, chỉ demo tính năng phản ánh trực tiếp Methodology.
- **Demo Query Live**: Chạy 1 câu hỏi quan hệ code phức tạp trên repo `llama-index-core`. Hiển thị RRF ranking, retrieved source paths/context và khả năng trace câu trả lời về source evidence.
- **Fallback**: Nếu Ollama chậm hoặc demo live fail, chuyển sang output JSON đã lưu sẵn và giải thích cùng luồng retrieval.
- *(Đẩy DB Inspection, Ingestion Live, Visual Portal sang Slides Backup. Chỉ chiếu nếu Giám khảo hỏi trong Q&A)*.

### 6. KẾT LUẬN (1 Phút)
- Tổng kết ngắn gọn 3 câu trả lời cho 3 RQ.
- Mở ra hướng nghiên cứu tiếp theo (Mở rộng Java/Go, Human Review, thêm Baselines).

------------------------------------------------------------------------------------------
# KỊCH BẢN PHẢN BIỆN (Q&A BACKUP — 15 Phút)

## Về phương pháp RRF
Q: Tại sao dùng RRF mà không dùng Reranker Model?
A: RRF hợp nhất thứ hạng (rank) chứ không cộng điểm thô (score) vì 2 DB khác thang đo. RRF không cần thêm model inference hoặc dữ liệu huấn luyện; nó chỉ thực hiện phép cộng theo thứ hạng nên overhead nhỏ so với learned reranker. Luận văn chưa thực nghiệm so sánh RRF với reranker.

Q: k=60 trong RRF — tối ưu hay chọn mặc định?
A: k=60 là giá trị mặc định từ bài báo gốc (Cormack et al. 2009). Chưa tune trên development set riêng. Adaptive RRF weights là hướng nghiên cứu tiếp theo.

## Về thực nghiệm & thống kê
Q: 30 câu hỏi ai viết? Có bias không?
A: Câu hỏi được viết dựa trên source code review của llama-index-core, phân theo 3 mức độ (simple/medium/hard). Gold answers được AI source review approve, chưa có independent human annotation.

Q: Vì sao dùng automated judge thay human evaluation?
A: Chạy hoàn toàn local (qwen2.5-coder:7b). Human review là hướng phát triển cần thiết để tăng độ tin cậy.

Q: RAGAS judge có được validate không?
A: RAGAS operationalizes ba metrics, nhưng local judge chưa được calibrated với independent human labels. Vì vậy kết quả là comparative automated evidence, không phải human-validated accuracy.

Q: Vì sao chỉ một repo là confirmatory workload?
A: Thesis đã khóa confirmatory workload ở một snapshot LlamaIndex để kiểm soát source identity, schema, embedding model và paired design. Transformers & LangChain là supplementary replication được chạy sau đó, không thay đổi conclusion của thesis đã nộp.

## Về kiến trúc & kỹ thuật
Q: Hệ thống có chạy được với Java/Go không?
A: Kiến trúc Hexagonal tách biệt tầng trích xuất. Hiện AST Parser chỉ xử lý Python. Java/Go cần tích hợp thêm Tree-sitter adapter. Đã implement Fail-closed boundary để từ chối an toàn.

Q: Entity resolution sai thì ảnh hưởng thế nào?
A: Resolver chỉ redirect khi tìm được đúng 1 target (unique match). Ambiguous/missing matches giữ nguyên stub node, tránh tạo edge sai. Hy sinh recall để bảo toàn precision.

Q: Provenance check kiểm tra những trường nào?
A: Source identity (commit hash), index run ID, schema version, embedding model. Cache generation bị invalidate sau mỗi lần re-index.

Q: Incremental indexing đã đo speedup chưa?
A: Cơ chế đã implement và unit-tested, nhưng chưa có controlled sequential-vs-incremental benchmark. Không claim speedup percentage.

## Về hiệu năng
Q: Thời gian sinh câu trả lời LLM local 87s có quá chậm?
A: Retrieval trung bình 371 ms; tổng thời gian khoảng 87 giây chủ yếu nằm ở local LLM generation trên phần cứng benchmark (Apple M4 mini). Đây là giới hạn triển khai hiện tại, không phải retrieval latency.

Q: Cross-repo results (Transformers/LangChain) có CI tương đương không?
A: Supplementary report hiện đã chạy 10 case pairs mỗi repo nhưng artifact đang checked-in chỉ báo cáo means/deltas, chưa dùng CI để kết luận quality hoặc generalization. Transformers có Correctness Δ +0.041; LangChain giảm Correctness trong slice này. Cả hai đều cho thấy retrieval overhead và prompt-token reduction. Đây là post-submission descriptive evidence, không phải pooled CI hay bằng chứng generalization.

## Về baseline coverage
Q: Vì sao chưa có BM25/CodeBERT/GraphRAG baseline?
A: Phạm vi luận văn tập trung vào so sánh Hybrid vs Vector-only trên cùng điều kiện. BM25, graph-only, CodeBERT, GraphRAG là hướng mở rộng cần thiết để định vị đầy đủ hơn.

Q: Java/Go fail-closed boundary đã test thế nào?
A: Unit test kiểm tra CLI reject Java input với explicit error trước khi bắt đầu indexing. Không có silent fallback.
