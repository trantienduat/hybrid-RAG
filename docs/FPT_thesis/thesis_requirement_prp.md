# Product Requirement Prompt (PRP) — FPT University Master Thesis Formatting & Structure

## 🎯 Role & Objective
You are an expert academic writer and LaTeX typesetter. Your task is to update and format the Master Thesis for **Tran Tien Doat** in both English ([thesis_report_en.tex](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/docs/FPT_thesis/my_thesis/thesis_report_en.tex)) and Vietnamese ([thesis_report_vi.tex](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/docs/FPT_thesis/my_thesis/thesis_report_vi.tex)) formats to strictly adhere to **FPT University's Master of Software Engineering (MSE)** graduation thesis templates.

---

## 🏛️ Format & Page Layout Guidelines
Based on the FPT University MSE sample theses:
1.  **Cover Page Layout**:
    *   **Header**: `MINISTRY OF EDUCATION AND TRAINING` / `BỘ GIÁO DỤC VÀ ĐÀO TẠO` followed by `FPT UNIVERSITY` / `TRƯỜNG ĐẠI HỌC FPT`.
    *   **Title**: Large, bold, centered, and fully capitalized.
    *   **Author**: `by [Name] - [Class]` (e.g. `by Tran Tien Doat - MSE Class`) / `Học viên: [Name] - Lớp: [Class]`.
    *   **Degree Submission Note**: `A thesis submitted in conformity with the requirements for the degree of Master of Software Engineering` / `Luận văn trình bày để đáp ứng các yêu cầu cho học vị Thạc sĩ Kỹ nghệ Phần mềm`.
    *   **Supervisor**: `Supervisor: [Advisor Name]` / `Người hướng dẫn khoa học: [Advisor Name]`.
    *   **Copyright**: `© Copyright by [Student Name] [Year]` / `© Bản quyền thuộc về [Student Name] [Year]`.
2.  **Margins & Spacing**:
    *   Left: 3.0 cm, Right: 2.0 cm, Top: 2.5 cm, Bottom: 2.5 cm.
    *   Line spacing: 1.5 lines.
3.  **Front Matter Pages**:
    *   Outer Cover Page (without supervisor).
    *   Inner Cover Page (with supervisor).
    *   Acknowledgements page (`Acknowledgments` / `Lời cảm ơn`).
    *   Thesis Abstract page (`Thesis Abstract` / `Tóm tắt Luận văn`).
    *   Table of Contents (`Table of Contents` / `Mục lục`).
    *   List of Tables (`List of Tables` / `Danh mục Bảng biểu`).
    *   List of Figures (`List of Figures` / `Danh mục Hình vẽ`).
    *   List of Abbreviations (`List of Abbreviations` / `Danh mục Từ viết tắt`).

---

## 🛠️ Structure & Section Requirements

### 1. Cover Page & Front Matter Updates
*   Replace **HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY** and **TRƯỜNG ĐẠI HỌC BÁCH KHOA** with **FPT UNIVERSITY** and **TRƯỜNG ĐẠI HỌC FPT**.
*   Update major from Computer Science (Khoa học Máy tính) to **Software Engineering** (Kỹ nghệ Phần mềm).
*   Add a formal **Acknowledgments** / **Lời cảm ơn** section before the Abstract.
*   Add macro definitions/environments for **List of Tables**, **List of Figures**, and **List of Abbreviations** to generate the lists automatically in LaTeX.

### 2. Thesis Content Refinements
Verify that the thesis chapters correspond to FPT University Software Engineering thesis guidelines:
*   **Chapter 1: Introduction**: Background, Problem Statement, Objectives, Contributions, Outline.
*   **Chapter 2: Literature Review**: RAG, Graph Databases (FalkorDB), Vector Databases (Qdrant), RRF Rank Fusion.
*   **Chapter 3: Proposed Architecture**: Core Ingestion, AST parsing (Tree-Sitter), Scoped Retrieval, RRF Merger.
*   **Chapter 4: System Implementation**: Folder layout, Ports & Adapters, OpenTelemetry, Arize Phoenix integration.
*   **Chapter 5: Experimental Results & Discussions**: Local testing setup (Mac mini M4, Ollama, Gemma), Hit Rate @5 comparisons, Ragas evaluations, latency benchmarks.
*   **Chapter 6: Conclusions & Future Work**: Key milestones, limitations, future extensions.
*   **References**: Formatted alphabetically or numerically.
