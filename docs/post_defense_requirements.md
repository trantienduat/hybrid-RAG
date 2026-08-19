# Ghi nhận ý kiến sau buổi bảo vệ

## Nguồn

- **Biên bản Hội đồng chấm Đề án Thạc sĩ:** họp ngày 19/08/2026.
- **Đề tài:** *Hybrid-RAG Code Retrieval System Combining Knowledge Graphs and Semantic Search*.
- **Học viên:** Trần Tiến Duật.
- **Các phiếu nhận xét:** TS Nguyễn Xuân Sâm, TS Nguyễn Ngọc Trường Minh và TS Hồ Hải Vân.

## I. Phần ghi nhận trong biên bản Hội đồng

Phần ghi tay tại mục “Các nội dung cần hoàn thiện (nếu có)” ghi:

> “Nghiêm túc sửa đổi báo cáo theo ý kiến Hội đồng.”

Các câu hỏi ghi tay đọc được gồm:

1. “Tại sao dùng RRF?”
2. Câu hỏi về khả năng phù hợp với các codebase khác.
3. “Độ trễ? 84s.”

Trong phần ghi tay về dataset có ghi chú về việc cập nhật thủ công và đồng bộ bằng tay. Chữ viết tay ở đoạn này không đủ rõ để xác nhận thêm nội dung ngoài ý chính trên.

## II. Nhận xét của TS Nguyễn Xuân Sâm

### Các hạn chế được nêu

- Đánh giá mới thực hiện trên một repository Python duy nhất và chỉ có một baseline được chạy trực tiếp là vector-only.
- Các baseline quan trọng như BM25, graph-only, code encoder hoặc một triển khai GraphRAG độc lập chưa được so sánh.
- Hệ thống indexing dựa trên AST hiện hỗ trợ Python, chưa thể khẳng định khả năng tổng quát cho Java, Go hoặc các repository đa ngôn ngữ.
- 30 câu hỏi có quy mô còn nhỏ; reference answer được AI rà soát, chưa phải đánh giá độc lập bởi chuyên gia.
- Thời gian sinh câu trả lời local trung bình khoảng 87--97 giây, còn cao đối với trải nghiệm tương tác.

### Hướng được nêu trong nhận xét

- Tiếp tục đánh giá trên nhiều repository.
- Bổ sung baseline và/hoặc ablation.
- Mở rộng adapter cho các ngôn ngữ khác.
- Tăng cường human evaluation.

## III. Nhận xét của TS Nguyễn Ngọc Trường Minh

### Những vấn đề cần bổ sung và sửa chữa

- **Bổ sung thực nghiệm:** thử nghiệm thêm trên Transformers, LangChain và ít nhất BM25, graph-only hoặc GraphRAG để tăng độ tin cậy và khả năng khái quát.
- **Cải thiện hình thức:** chuẩn hóa kích thước chữ, khoảng cách, ký hiệu và tăng tính trực quan cho kiến trúc hệ thống.
- **Tài liệu tham khảo:** bổ sung các nghiên cứu giai đoạn 2025--2026 về code-RAG, repository-level retrieval và agentic software engineering.

### Các câu hỏi được ghi trong phiếu

1. Tại sao sử dụng Reciprocal Rank Fusion thay vì chuẩn hóa trực tiếp điểm similarity của graph retrieval và vector retrieval rồi cộng có trọng số?
2. Kết quả Hybrid-RAG cao hơn vector-only ở cả ba chỉ số. Có thể kết luận Hybrid-RAG tốt hơn một cách có ý nghĩa thống kê hay không? Tại sao?
3. Đề án chỉ thử nghiệm trên một phiên bản LlamaIndex Core và Python. Dựa trên cơ sở nào có thể cho rằng kiến trúc này phù hợp với các codebase doanh nghiệp khác?
4. Generation latency trung bình khoảng 87 giây với Hybrid-RAG. Với thời gian này, hệ thống có thực sự khả thi cho developer sử dụng tương tác trong môi trường production hay không?

## IV. Nhận xét của TS Hồ Hải Vân

### Những vấn đề cần bổ sung và sửa chữa

- Cần tăng quy mô tập câu hỏi và có human evaluation độc lập. Reference answers hiện được AI review, chưa phải annotation độc lập của chuyên gia.
- Với `N=30`, Answer Correctness có cận dưới của CI chỉ vừa vượt zero ở khoảng `0.002`, nên bằng chứng thống kê vẫn tương đối mong manh.
- Phần Metrics giới thiệu Hit Rate@5 nhưng không báo cáo một kết quả Hit Rate@5 hợp lệ từ benchmark schema-v3; báo cáo structural 50-case cũ đã bị loại vì provenance không đạt yêu cầu. Cần bổ sung Hit Rate@5 hiện hành hoặc bỏ metric này khỏi nhóm chỉ số đánh giá chính.
- Cần bổ sung phân tích theo `simple/medium/hard` và loại câu hỏi `structural/semantic`.
- Cần ablation các trọng số RRF và power analysis.
- Retrieval latency tăng từ `134` lên `371 ms`, trong khi generation mất `87--97 giây`; cần phân biệt rõ “retrieval SLA <1 giây” với trải nghiệm end-to-end của người sử dụng.

## V. Phạm vi ghi nhận

- File này chỉ ghi nhận nội dung có trong PDF gốc, theo biên bản và từng phiếu nhận xét.
- PDF không nêu thứ tự ưu tiên hoặc mức độ khó giữa các yêu cầu; vì vậy không tự sắp xếp nhẹ đến nặng.
- Không tự suy ra chapter, model, dataset, baseline, ablation hoặc task triển khai ngoài các nội dung được nêu ở trên.
- Không ghi thêm kết luận “không cần thay đổi kiến trúc chính” vì PDF không có câu này; PDF chỉ nêu các hạn chế và yêu cầu bổ sung/sửa chữa.

## VI. Bảng giải trình cho các điều chỉnh trong thesis English

| Nội dung góp ý | Giải trình cho điều chỉnh tương ứng |
|---|---|
| **Hội đồng:** Nghiêm túc sửa đổi báo cáo theo ý kiến Hội đồng. | Thesis English được điều chỉnh ở các điểm có thể kiểm chứng từ evidence hiện tại: lý do chọn RRF, cách đọc 3 metric và CI, provenance/dataset, Hit Rate@5, phạm vi tổng quát hóa và latency. Các thực nghiệm chưa chạy không được trình bày như kết quả đã hoàn thành. |
| **TS Nguyễn Ngọc Trường Minh:** Vì sao dùng RRF thay vì chuẩn hóa similarity rồi cộng có trọng số? | Bổ sung giải thích rằng graph score và vector similarity không có thang đo chung được bảo đảm; weighted score fusion sẽ cần thêm quy tắc normalization/calibration. RRF tránh giả định này, còn weight biểu diễn mức ưu tiên của từng channel. Thesis cũng ghi rõ weighted-sum là alternative chưa được benchmark, nên không tuyên bố RRF luôn tốt hơn. |
| **TS Nguyễn Ngọc Trường Minh:** Kết quả cao hơn ở cả ba metric có ý nghĩa thống kê không? | Thesis giữ nguyên và trình bày đầy đủ Faithfulness, Answer Relevancy và Answer Correctness cùng paired case-level 95% CI. Chỉ CI của Answer Correctness hoàn toàn lớn hơn zero; Faithfulness không kết luận được và Answer Relevancy borderline. |
| **TS Hồ Hải Vân:** Hit Rate@5 được giới thiệu nhưng chưa có kết quả schema-v3 hợp lệ. | Hit Rate@5 được hạ xuống diagnostic metric definition. Báo cáo structural schema-v2 cũ tiếp tục bị loại khỏi confirmatory evidence vì provenance không hợp lệ; thesis không thêm một con số Hit Rate@5 chưa có evidence. |
| **Biên bản và nhận xét về dataset:** Dataset được cập nhật/thao tác thủ công và cần làm rõ cách tạo, cập nhật, đồng bộ. | Thesis mô tả gold dataset là checked-in artifact được duy trì thủ công, review trên pinned source snapshot và ràng buộc bằng wheel/source digest cùng index provenance. Đồng thời ghi rõ đây vẫn là limitation và không thay thế human annotation độc lập. |
| **TS Nguyễn Xuân Sâm, TS Nguyễn Ngọc Trường Minh và TS Hồ Hải Vân:** Một repository Python, một baseline đã chạy; cần thận trọng với claim về codebase khác, ngôn ngữ khác và doanh nghiệp. | Thesis giới hạn kết luận vào pinned LlamaIndex Core/Python workload và vector-only baseline. Transformers, LangChain, BM25, graph-only, GraphRAG, cross-language và multi-repository generalization được ghi là chưa có evidence hoặc future work, không suy diễn thành kết quả. |
| **TS Nguyễn Xuân Sâm và TS Nguyễn Ngọc Trường Minh:** Generation khoảng 87--97 giây; cần đánh giá khả năng dùng tương tác. | Thesis tách retrieval latency 134/371 ms khỏi generation latency khoảng 87--97 giây, đồng thời giải thích end-to-end experience còn bao gồm context assembly và prompt processing. Vì vậy chỉ báo cáo sub-second mean retrieval trên máy đo, không tuyên bố sub-second answer SLA. |
| **TS Hồ Hải Vân:** Cần tăng câu hỏi, human evaluation, phân tích structural/semantic, ablation RRF weights và power analysis. | Thesis hiện đã giữ phân tầng 10 simple/10 medium/10 hard và paired bootstrap trên 30 case. Human evaluation, structural/semantic analysis, RRF ablation và power analysis chưa có kết quả trong evidence hiện tại nên được giữ ở limitations/future work, không ghi như việc đã hoàn thành. |
| **TS Nguyễn Ngọc Trường Minh:** Bổ sung Transformers, LangChain, BM25, graph-only/GraphRAG; cập nhật hình thức và tài liệu tham khảo 2025--2026. | Thesis không thêm claim hoặc citation chưa được kiểm chứng. Các benchmark/baseline còn thiếu tiếp tục được ghi là not executed; yêu cầu refresh tài liệu tham khảo 2025--2026 được thêm vào future work. Phần chỉnh hình thức/sơ đồ được giữ như một hạng mục hoàn thiện báo cáo riêng, không giả định đã hoàn tất chỉ từ việc cập nhật nội dung này. |
