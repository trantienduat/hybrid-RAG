#!/usr/bin/env python3
"""Validate one-to-one structural parity between Vietnamese and English theses."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VI_PATH = ROOT / "docs/thesis_report.tex"
EN_PATH = ROOT / "docs/thesis_report_en.tex"

# The Vietnamese report intentionally contains an additional Vietnamese abstract.
VI_ONLY_FRONT_MATTER = {"Tóm tắt Luận văn"}

# Ordered map: (heading level, Vietnamese title, English title).
EXPECTED_PAIRS = [
    ("chapter*", "Abstract", "Abstract"),
    ("chapter", "Mở đầu", "Introduction"),
    ("section", "Ngữ cảnh hệ thống (Context)", "System Context"),
    (
        "section",
        "Đặc tả bài toán và Điểm nghẽn (Problem Statement)",
        "Problem Statement and Bottlenecks",
    ),
    ("section", "Đánh giá các giải pháp hiện hành (Related Works)", "Related Works"),
    ("section", "Giải pháp đề xuất (Proposed Solution)", "Proposed Solution"),
    (
        "section",
        "Khoảng trống nghiên cứu và Đóng góp kỹ thuật",
        "Research Gap and Technical Contributions",
    ),
    (
        "section",
        r"Mục tiêu và Câu hỏi nghiên cứu (Research Objectives \& Questions)",
        r"Research Objectives \& Questions",
    ),
    (
        "subsection",
        "Mục tiêu nghiên cứu (Research Objectives)",
        "Research Objectives",
    ),
    (
        "subsection",
        "Câu hỏi nghiên cứu (Research Questions)",
        "Research Questions",
    ),
    (
        "section",
        r"Phạm vi nghiên cứu và Tiêu chí hoàn thành (Scope \& DoD)",
        r"Research Scope \& Definition of Done",
    ),
    ("subsection", "Phạm vi nghiên cứu (Scope)", "Research Scope"),
    (
        "subsection",
        "Tiêu chí hoàn thành (Definition of Done - DoD)",
        "Definition of Done (DoD)",
    ),
    (
        "chapter",
        "Cơ sở lý thuyết và Kiến trúc hệ thống",
        "Theoretical Background and System Architecture",
    ),
    (
        "section",
        "Cơ sở lý thuyết và Tổng quan tài liệu",
        r"Theoretical Background \& Literature Review",
    ),
    (
        "subsection",
        "Lý thuyết Truy xuất Thông tin (Information Retrieval Theory)",
        "Information Retrieval (IR) Theory",
    ),
    ("subsection", "Các mô hình tìm kiếm mã nguồn (Code Search Models)", "Code Search Models"),
    (
        "subsection",
        "Thế hệ truy xuất tăng cường (RAG) và Graph RAG",
        "Retrieval-Augmented Generation (RAG) and Graph RAG",
    ),
    (
        "subsection",
        "Tổng hợp tài liệu và Hàm ý thiết kế",
        "Literature Synthesis and Design Implications",
    ),
    ("section", "Kiến trúc hệ thống", "System Architecture"),
    (
        "section",
        "Đặc tả yêu cầu hệ thống (System Requirements)",
        "System Requirements",
    ),
    (
        "section",
        "Kiến trúc tổng quan (High-level Architecture)",
        "High-level Architecture",
    ),
    (
        "section",
        "Đường ống nạp dữ liệu chi tiết (Ingestion Pipeline)",
        "Ingestion Pipeline",
    ),
    (
        "subsection",
        r"Cơ chế nạp dữ liệu tăng dần và đồng bộ (Incremental Indexing \& Synchronization)",
        "Incremental Indexing and Synchronization",
    ),
    (
        "subsection",
        "Hỗ trợ cấu trúc Mono-repo và thư mục con",
        "Monorepo and Subdirectory Support",
    ),
    (
        "section",
        "Kiến trúc truy xuất lai (Hybrid Retrieval Architecture)",
        "Hybrid Retrieval Flow",
    ),
    ("chapter", "Thiết kế chi tiết", "Detailed Design"),
    ("section", "Mô hình hóa dữ liệu (Database Schema)", "Database Schema"),
    ("section", "Thuật toán cốt lõi (Core Algorithms)", "Core Algorithms"),
    (
        "subsection",
        "Thuật toán giải quyết thực thể toàn cục (Global Entity Resolution)",
        "Global Entity Resolution",
    ),
    (
        "subsection",
        "Thuật toán xếp hạng trộn lai Reciprocal Rank Fusion (RRF)",
        "Reciprocal Rank Fusion (RRF)",
    ),
    (
        "section",
        "Phân vùng cộng đồng và Tổng hợp kiến trúc",
        "Community Detection and Architectural Clustering",
    ),
    (
        "subsection",
        "Phân vùng theo thư mục và Sinh tóm tắt",
        "Directory-Based Partitioning and Summarization",
    ),
    (
        "subsection",
        "Tích hợp với Đường ống truy xuất",
        "Integration with the Retrieval Pipeline",
    ),
    (
        "section",
        "Đồ thị đa repository và Cổng trực quan hóa",
        "Multi-Repository Knowledge Graph and Visualization Portal",
    ),
    ("subsection", "Master Graph đa repository", "Multi-Repository Master Graph"),
    (
        "subsection",
        "Điều hướng Drill-Down phân cấp",
        "Hierarchical Drill-Down Navigation",
    ),
    (
        "subsection",
        "Tối ưu xử lý đồng thời cho Indexing",
        "Performance Optimizations for Concurrent Indexing",
    ),
    (
        "section",
        "Khả năng mở rộng theo ngôn ngữ lập trình",
        "Programming-Language Scalability",
    ),
    ("section", "Kiểm thử phần mềm (Software Testing)", "Software Testing"),
    ("chapter", "Kết quả thực nghiệm và Thảo luận", "Experimental Results and Discussion"),
    ("section", "Thiết lập môi trường kiểm thử (Test Environment)", "Test Environment"),
    (
        "section",
        "Lý do lựa chọn thang đo đánh giá (Metrics Justification)",
        "Metrics Justification",
    ),
    (
        "section",
        "Provenance bằng chứng và Diagnostic lịch sử bị loại",
        "Evidence Provenance and Excluded Historical Diagnostic",
    ),
    (
        "section",
        "Benchmark chất lượng câu trả lời độc lập",
        "Independent Answer-Quality Benchmark",
    ),
    (
        "section",
        "Khả năng tổng quát và Phạm vi baseline",
        "Generalizability and Baseline Coverage",
    ),
    (
        "section",
        "Đánh giá và tối ưu hóa kiến trúc hệ thống",
        "Architectural Optimization and Enhancements",
    ),
    (
        "section",
        "Đo lường thời gian xử lý chi tiết (Latency Breakdown)",
        "Latency Breakdown",
    ),
    ("section", "Nghiên cứu tình huống thực tế (Case Study)", "Case Study"),
    ("chapter", "Kết luận và Hướng phát triển", "Conclusion and Future Work"),
    ("section", "Kết luận", "Conclusion"),
    ("section", "Hạn chế kỹ thuật (Limitations)", "Limitations"),
    ("section", "Hướng phát triển tương lai (Future Work)", "Future Work"),
]

HEADING_RE = re.compile(
    r"^\\(?P<level>chapter|section|subsection|subsubsection)(?P<star>\*?)"
    r"\{(?P<title>[^{}]*)\}",
    re.MULTILINE,
)
SEMANTIC_ENVS = {"table", "figure", "equation", "enumerate", "itemize"}


def _normalize_heading_markup(text: str, *, vi: bool) -> str:
    """Map layout-only English front-matter markup to its semantic heading."""
    if not vi:
        return text.replace(
            r"\frontmatterheading{Abstract}",
            r"\chapter*{Abstract}",
        )
    return text


def _headings(text: str, *, vi: bool) -> list[tuple[str, str]]:
    text = _normalize_heading_markup(text, vi=vi)
    headings = []
    for match in HEADING_RE.finditer(text):
        level = f"{match.group('level')}{match.group('star')}"
        title = match.group("title")
        if vi and title in VI_ONLY_FRONT_MATTER:
            continue
        headings.append((level, title))
    return headings


def _environment_sequence(text: str) -> list[str]:
    return [env for env in re.findall(r"\\begin\{([^{}]+)\}", text) if env in SEMANTIC_ENVS]


def _block_signatures(text: str, *, vi: bool) -> list[tuple[int, ...]]:
    """Count semantic elements inside each mapped heading block."""
    text = _normalize_heading_markup(text, vi=vi)
    matches = list(HEADING_RE.finditer(text))
    signatures = []
    for index, match in enumerate(matches):
        if vi and match.group("title") in VI_ONLY_FRONT_MATTER:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        signatures.append(
            (
                block.count(r"\item"),
                block.count(r"\begin{table}"),
                block.count(r"\begin{figure}"),
                block.count(r"\begin{equation}"),
                block.count(r"\begin{enumerate}"),
                block.count(r"\begin{itemize}"),
            )
        )
    return signatures


def _labels(text: str) -> set[str]:
    return set(re.findall(r"\\label\{([^{}]+)\}", text))


def _citations(text: str) -> set[str]:
    citations: set[str] = set()
    for group in re.findall(r"\\cite\{([^{}]+)\}", text):
        citations.update(key.strip() for key in group.split(","))
    return citations


def _bibitems(text: str) -> set[str]:
    return set(re.findall(r"\\bibitem\{([^{}]+)\}", text))


def main() -> None:
    vi_text = VI_PATH.read_text(encoding="utf-8")
    en_text = EN_PATH.read_text(encoding="utf-8")
    actual_vi = _headings(vi_text, vi=True)
    actual_en = _headings(en_text, vi=False)
    expected_vi = [(level, vi_title) for level, vi_title, _ in EXPECTED_PAIRS]
    expected_en = [(level, en_title) for level, _, en_title in EXPECTED_PAIRS]

    errors = []
    if actual_vi != expected_vi:
        errors.append("Vietnamese heading order differs from the explicit parity map")
    if actual_en != expected_en:
        errors.append("English heading order differs from the explicit parity map")
    if _environment_sequence(vi_text) != _environment_sequence(en_text):
        errors.append("Table/figure/equation/list structure differs")
    if _block_signatures(vi_text, vi=True) != _block_signatures(en_text, vi=False):
        errors.append("Mapped sections contain different numbers of items/tables/figures/equations")
    if _labels(vi_text) != _labels(en_text):
        errors.append("LaTeX label sets differ")
    if _citations(vi_text) != _citations(en_text):
        errors.append("Citation sets differ")
    vi_bibitems = _bibitems(vi_text)
    en_bibitems = _bibitems(en_text)
    if vi_bibitems != en_bibitems:
        errors.append("Bibliography key sets differ")
    undefined_vi = _citations(vi_text) - vi_bibitems
    undefined_en = _citations(en_text) - en_bibitems
    if undefined_vi:
        errors.append(f"Vietnamese thesis has undefined citations: {sorted(undefined_vi)}")
    if undefined_en:
        errors.append(f"English thesis has undefined citations: {sorted(undefined_en)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    for index, (level, vi_title, en_title) in enumerate(EXPECTED_PAIRS, start=1):
        print(f"{index:02d} {level}: {vi_title} <-> {en_title}")
    print(
        f"PASS: {len(EXPECTED_PAIRS)} mapped headings; "
        "semantic environments, labels, citations, and bibliography keys are aligned"
    )


if __name__ == "__main__":
    main()
