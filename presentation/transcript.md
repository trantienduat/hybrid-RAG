## Slide 1 — Code Retrieval with Hybrid-RAG

“Luận văn của em nghiên cứu bài toán truy xuất mã nguồn. Vấn đề là nhiều câu hỏi về code không thể trả lời tốt chỉ bằng việc tìm các đoạn văn có từ khóa tương
tự.

Hệ thống của em kết hợp hai loại thông tin: semantic search để tìm code có ý nghĩa gần với câu hỏi, và knowledge graph được trích xuất từ AST để biểu diễn các
quan hệ như CALLS, IMPORTS, INHERITS.

Em không claim rằng mình phát minh ra một thuật toán retrieval hoàn toàn mới. Đóng góp chính là tích hợp các thành phần này thành một hệ thống code retrieval
có phạm vi repository, có provenance và có thể kiểm tra được bằng thực nghiệm.”

Điểm hội đồng cần hiểu: đây là một hệ thống tích hợp và đánh giá có kiểm soát, không phải claim về một primitive mới.

## Slide 2 — A code question is often a relationship question

“Câu hỏi ví dụ là: BaseRetriever.retrieve thực hiện chuỗi xử lý nào khi nhận một query string, bao gồm callback và recursive retrieval?

Vector search có thể tìm thấy các file chứa từ retrieve, callback hoặc query. Tuy nhiên, nó không trực tiếp biểu diễn được phương thức nào gọi phương thức
nào, callback xuất hiện ở bước nào, hay recursive retrieval liên quan ra sao.

Vì vậy, semantic retrieval giúp tìm đúng khu vực của code, còn structural graph giúp giải thích quan hệ thực thi giữa các thành phần.”

Điểm chốt: semantic similarity tìm được “đoạn code liên quan”, nhưng chưa chắc giải thích được “mối quan hệ giữa các đoạn code”.

## Slide 3 — Three research questions

“Luận văn đặt ra ba câu hỏi nghiên cứu.

RQ1 hỏi về answer quality: Hybrid-RAG có cải thiện Faithfulness, Answer Relevancy và Answer Correctness so với vector-only hay không?

RQ2 hỏi về efficiency trade-off: việc thêm graph retrieval ảnh hưởng thế nào đến retrieval latency, generation latency và token footprint?

RQ3 hỏi về evidence strength: các khác biệt quan sát được có còn nằm trên hoặc dưới zero khi tính paired 95% confidence intervals hay không?

RQ3 rất quan trọng vì một point estimate dương chưa đủ để kết luận rằng hệ thống thực sự cải thiện.”

## Slide 4 — The experiment is controlled—but deliberately narrow

“Thực nghiệm của em được kiểm soát vì hai điều kiện vector-only và hybrid sử dụng cùng source snapshot, cùng bộ câu hỏi, cùng answer model, cùng judge và cùng
số lần chạy.

Tuy nhiên, phạm vi thực nghiệm cố ý hẹp: chỉ có một repository Python được pin là llama-index-core==0.14.21, một baseline vector-only và reference answers
được AI source-review.

BM25, CodeBERT, GraphCodeBERT và Microsoft GraphRAG được dùng để positioning trong literature, chứ chưa phải các baseline đã chạy trong thực nghiệm này.

Vì vậy, claim hợp lệ là Hybrid-RAG tốt hơn trên workload được pin và ghép cặp này, không phải tốt hơn mọi hệ thống RAG trong mọi repository.”

Nếu hội đồng hỏi “So với GraphRAG/BM25 thì sao?”, trả lời:

“Trong thesis này, đó là literature comparison, chưa phải experimental baseline. Experimental comparison chỉ gồm Hybrid-RAG và vector-only trong cùng điều
kiện.”

## Slide 5 — One indexed source supports three query-aware retrieval paths

“Ở index time, cùng một source snapshot được chuyển thành hai representation: graph representation và vector representation. Hai representation này dùng chung
source identity, schema version, embedding model và index run ID.

Ở query time, hệ thống không xử lý mọi câu hỏi theo cùng một cách.

Exact-relation route dùng graph plan cho những câu hỏi cần quan hệ cụ thể. Generic-local route chạy graph và vector retrieval rồi fusion kết quả. Global route
sử dụng community summaries để trả lời các câu hỏi tổng quan về kiến trúc repository.

Do đó, Hybrid-RAG ở đây là query-aware retrieval, không phải lúc nào cũng đơn giản là chạy graph và vector song song.”

## Slide 6 — One function becomes two different kinds of evidence

“Một function trong hệ thống được biểu diễn dưới hai dạng.

Graph lưu các quan hệ rõ ràng, ví dụ module định nghĩa class nào, function nào gọi function nào, hoặc module nào import module nào.

Vector store lưu source text và semantic meaning của đoạn code. Nó phù hợp để tìm các đoạn code liên quan về mặt ngữ nghĩa.

Hai representation này bổ sung cho nhau: vector search giúp tìm vùng code phù hợp, còn graph traversal giúp nối các vùng đó thành một explanation có cấu
trúc.”

Nếu hội đồng hỏi “Graph có thay thế vector không?”, trả lời:

“Không. Graph mạnh ở explicit relation, nhưng vector mạnh ở semantic matching. Thiết kế của em giữ cả hai vì hai loại evidence giải quyết hai phần khác nhau
của câu hỏi.”

## Slide 7 — Generic local queries fuse complementary rankings

“Trong generic local route, điểm graph và điểm vector không cùng thang đo, nên hệ thống không cộng trực tiếp raw scores.

Thay vào đó, RRF sử dụng vị trí xếp hạng của mỗi candidate. Với generic local route, graph weight được giới hạn ở 1.0 và vector weight là 1.0. Sau khi fusion,
hệ thống deduplicate các node và giữ context phù hợp để gửi cho local LLM.

Với exact relation route, graph có thể được ưu tiên mạnh hơn. Còn trong benchmark answer quality, hệ thống giữ năm context items đầu tiên.

Điểm cần nhấn mạnh là RRF kết hợp rank positions, không giả định graph score và vector score đã được calibration trên cùng một scale.”

Nếu bị hỏi “Tại sao k bằng 60?”, trả lời:

“60 là smoothing constant được dùng trong công thức RRF hiện tại. Trong thesis này, em không claim đây là giá trị tối ưu; đó là một cấu hình cố định được giữ
nhất quán trong thực nghiệm.”

## Slide 8 — Unsupported states fail at explicit boundaries

“Reliability của hệ thống không chỉ là trả về answer khi mọi thứ đúng, mà còn phải fail rõ ràng khi input hoặc state không được hỗ trợ.

Hệ thống reject Java vì semantic Java extraction chưa được triển khai. Nó kiểm tra repository identity, source digest, schema và embedding model trước khi
retrieval hoặc evaluation.

Trong quá trình indexing, dữ liệu mới được chuẩn bị trước khi database mutation, và records mới được ghi trước khi xóa records cũ. Điều này làm giảm empty-
index window, nhưng em không claim đây là reader-isolated atomic cutover.

Cache generation cũng thay đổi theo index generation để tránh việc answer từ snapshot cũ bị xem như answer hiện tại. Latest validation hiện có 315 tests
passed, 27 integration cases skipped rõ ràng và 2 warnings.”

Nếu hội đồng hỏi “Replacement-before-cleanup có atomic không?”, trả lời:

“Không hoàn toàn atomic đối với readers. Nó chỉ giảm khoảng thời gian index bị rỗng; đây là reliability improvement, không phải transactional cutover.”

## Slide 9 — The confirmatory workload preserves pairing

“Confirmatory benchmark có 30 câu hỏi, chia thành 10 simple, 10 medium và 10 hard.

Mỗi câu hỏi chạy trong hai điều kiện: vector-only và hybrid. Mỗi điều kiện chạy ba lần, nên tổng cộng có 180 raw records.

Tuy nhiên, 180 records không phải 180 independent observations. Ba lần chạy của cùng một câu hỏi được average trong từng mode. Sau đó, mỗi câu hỏi tạo thành
một pair giữa hybrid và vector-only. Vì vậy, đơn vị suy luận thật sự là 30 paired question differences.

Cuối cùng, em resample 30 pairs này 10,000 lần với seed 42 để tính paired 95% confidence intervals. Cách này tránh việc xem ba lần generation của cùng một câu
hỏi như ba bằng chứng độc lập.”

## Slide 10 — All point estimates favor Hybrid

“Ở mức point estimate, cả ba answer-quality metrics đều cao hơn khi dùng Hybrid-RAG.

Faithfulness tăng từ 0.896 lên 0.924, với delta khoảng +0.029, nhưng CI là [-0.069, 0.115], nên chưa thể kết luận chắc chắn.

Answer Relevancy tăng từ 0.729 lên 0.808, delta +0.079, nhưng CI vẫn chạm hoặc thấp hơn zero rất nhẹ.

Answer Correctness tăng từ 0.652 lên 0.698, delta +0.046, với CI [0.002, 0.094]. Đây là metric duy nhất có khoảng tin cậy hoàn toàn trên zero trong workload
này.

Các scores này là automated judge scores, không phải tỷ lệ phần trăm correctness khách quan do human adjudication đo.”

## Slide 11 — Positive average and reliable improvement

“Slide này phân biệt hai khái niệm.

Positive average chỉ nói rằng trong sample hiện tại, Hybrid có mean cao hơn. Reliable improvement yêu cầu confidence interval của delta không bao gồm zero.

Faithfulness vẫn tương thích với cả một hiệu ứng âm nhỏ và một hiệu ứng dương lớn, nên em gọi là inconclusive.

Answer Relevancy gần như hoàn toàn dương nhưng lower bound vẫn hơi dưới zero, nên em gọi là borderline.

Chỉ Answer Correctness có interval hoàn toàn dương. Vì vậy, em chỉ đưa ra claim mạnh cho correctness, còn hai metric kia được báo cáo trung thực là chưa đủ
evidence.”

## Slide 12 — Efficiency trade-off

“Hybrid retrieval làm retrieval latency tăng từ khoảng 134 ms lên 371 ms. Delta là khoảng +237 ms, với CI [208, 266], nên retrieval overhead này được dữ liệu
hỗ trợ.

Generation latency lại giảm từ khoảng 96.8 giây xuống 87.3 giây, nhưng CI là [-23.0, +1.4], vẫn chứa zero.

Prompt tokens giảm khoảng 101 tokens và completion tokens giảm khoảng 100 tokens, nhưng hai khoảng tin cậy này cũng chứa zero.

Vì vậy, kết luận chính về efficiency là Hybrid có retrieval overhead được đo rõ ràng. Em không claim rằng nó đã chứng minh generation hoặc token efficiency.
Con số 371 ms cũng chỉ là retrieval latency trong workload này, không phải end-to-end SLA.”

## Slide 13 — Focused live demo

“Demo chỉ minh họa một flow hoàn chỉnh, không dùng để thay thế benchmark.

Đầu tiên, em kiểm tra FalkorDB, Qdrant và Ollama đang hoạt động. Sau đó, em gửi một câu hỏi về execution sequence của BaseRetriever.retrieve, có repository
scope.

Em cho hội đồng xem graph candidates, vector candidates, RRF result, context được chọn và answer kèm source paths.

Cuối cùng, em thử indexing một Java fixture để cho thấy hệ thống reject unsupported input thay vì giả vờ index thành công.

Demo chứng minh mechanism có thể quan sát được; benchmark mới là cơ sở cho aggregate claim.”

## Slide 14 — The contribution is a trustworthy code-retrieval system

“Luận văn không claim phát minh AST parsing, graph retrieval, vector search hay RRF.

Đóng góp chính là tích hợp các thành phần đó thành một hệ thống code retrieval local, repository-scoped và auditable.

Hệ thống có query-aware routing, graph và vector indexes dùng chung provenance identity, có community summaries, fail-closed language boundary và evaluation
artifacts có thể kiểm tra lại.

Các chức năng như incremental indexing, Web UI, MCP server hay exploration portal là engineering capabilities. Chúng giúp hệ thống hữu dụng hơn, nhưng không
nên bị trình bày như các research claims đã được benchmark đầy đủ.”

## Slide 15 — Conclusion

“Kết luận chính cần giữ thật chính xác.

Trên workload LlamaIndex được pin, Hybrid-RAG cải thiện judged Answer Correctness với delta +0.046 và CI [0.002, 0.094].

Cái giá đo được là thêm khoảng 237 ms retrieval latency.

Faithfulness, Relevancy, generation latency và token reductions có point estimate tích cực hoặc thuận lợi ở một số khía cạnh, nhưng confidence intervals chưa
đủ để claim chắc chắn.

Vì vậy, conclusion của thesis là Hybrid-RAG cho thấy lợi ích về answer correctness trong workload được kiểm soát này, chứ chưa phải bằng chứng về
generalization cho mọi repository, mọi ngôn ngữ hoặc mọi baseline.”

## Các depth slides — chỉ dùng khi hội đồng hỏi

Slide 16 — Historical 50-case result: Đây là kết quả cũ dùng để diagnostic regression. Nó thuộc schema-two, trước khi provenance identity được sửa, nên không
được trộn vào confirmatory result.

Slide 17 — Routing decisions: Exact relation dùng graph plan; generic local dùng graph và vector concurrent; global dùng community summaries.

Slide 18 — Directory communities: Community được tạo theo repository và directory để hỗ trợ architectural questions. Đây không phải Louvain clustering.

Slide 19 — Graph schema: Graph phân biệt ownership hierarchy với dependency edges: directory–module–class–function, cùng với calls, imports và inheritance.

Slide 20 — Entity resolution: Resolver ưu tiên matching chắc chắn. Nếu call target không đủ rõ, nó giữ lại stub thay vì tự đoán quan hệ sai.

Slide 21 — RRF parameters: RRF dùng smoothing constant 60. Graph weight thay đổi theo route; vector-only đặt graph contribution bằng zero.

Slide 22 — Provenance preflight: Evaluation chỉ được chạy khi source digest, repository, schema, embedding model và index metadata khớp nhau.

Slide 23 — Immutable package identity: Benchmark gắn với một package artifact và source digest cụ thể, giúp tránh việc source version mới bị trộn với index
cũ.

Slide 24 — Evidence credibility: Reference answers được source-review và records có thể audit, nhưng chưa có independent human adjudication. Vì vậy không gọi
đây là objective production accuracy.

Slide 25 — Bootstrap analysis: Repeats của cùng một câu hỏi được average trước. Bootstrap resample 30 paired cases, không resample 180 raw answers.

Slide 26 — Defect controls: Các failure modes đều có contract rõ: repository filtering, provenance abort, cache generation, replacement-before-cleanup và Java
rejection.

Slide 27 — System breadth: Prototype còn có incremental indexing, bounded concurrency, community summaries và exploration portal. Đây là system capabilities,
không phải các efficiency claims đã được chứng minh.

Slide 28 — Future experiments: Future work gồm human adjudication, nhiều câu hỏi độc lập hơn, benchmark lớn hơn cho Transformers/LangChain, thêm baseline
BM25/graph-only/GraphRAG, Java extractor và resource measurements.

Cách nhớ toàn bộ presentation:

> Problem → Architecture → Experimental design → Results → Statistical interpretation → Trade-off → Demo → Precise conclusion.

Ba câu cần tuyệt đối giữ nhất khi trả lời hội đồng:

1. “The strongest confirmed result is Answer Correctness, not all three quality metrics.”
2. “The claim is workload-specific, not universal generalization.”
3. “Transformers and LangChain are supplementary source-anchored slices, not part of the thesis’s confirmatory conclusion.”