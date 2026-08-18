# Hybrid-RAG Presentation Style and Running-Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the thesis-defense HTML showcase and PPTX share one academic-editorial visual system and one visible `BaseRetriever.retrieve` running-case trace.

**Architecture:** Preserve the existing 15-core plus supporting-detail narrative and update only the visual language and explanatory connective tissue. The HTML will use shared CSS tokens and semantic diagram classes; the PPTX generator will use matching color/type/shape tokens and speaker-note wording. The running case will be an explanatory trace, not a replacement for the aggregate benchmark evidence.

**Tech Stack:** Self-contained HTML/CSS/JavaScript; `@oai/artifact-tool` JavaScript ESM PPTX generator; artifact-tool PNG/layout export; PowerPoint import/render verification.

## Global Constraints

- Keep the submitted RQs, benchmark values, confidence intervals, scope boundaries and evidence claims unchanged.
- Keep the current interleaved core/detail slide order.
- Use `#F2EFE8` paper, `#FBFAF7` deep surface, `#FFFDF8` panel, `#1D2A34` ink, `#586671` muted, `#2D6F78` graph, `#B65C3C` vector/process, `#3D6F58` evidence, and `#F4E2DA` limitation tokens in both artifacts.
- Use Aptos Display/Aptos-compatible sans typography for both artifacts; reserve monospace for code, identifiers and numeric evidence.
- Mark the example as representative trace; do not imply that one query proves the benchmark result.
- Preserve unrelated user modifications in `src/` and `tests/`; do not clean `.worktrees/` or `presentation/transcript.md`.

### Task 1: Normalize the shared visual language in the HTML showcase

**Files:**
- Modify: `presentation/hybrid_rag_thesis_showcase.html`

**Interfaces:**
- Consumes: existing slide markup, SVG diagrams, CSS custom properties and keyboard navigation.
- Produces: consistent light academic-editorial styling and reusable running-case marker classes.

- [x] Update root CSS tokens and typography so headings/body/code use the same semantic roles as the PPTX.
- [x] Replace leftover dark diagram surfaces with the light panel/semantic fills while retaining teal graph, orange vector/process, green evidence and coral limitation meaning.
- [x] Normalize borders, corner radii, arrow treatment, table numerals and footer/kicker spacing without changing slide content hierarchy.
- [x] Add a reusable `running-case` marker and place it on the problem slide, retrieval/method slides, evaluation slide, contribution slide and conclusion slide.
- [x] Add explicit sibling-route wording for communities: the current local query does not traverse the global/community route.
- [x] Keep notes/source blocks intact and ensure narrow-screen fallback still works.

### Task 2: Apply the same tokens and running-case treatment to the PPTX generator

**Files:**
- Modify: `/private/tmp/hybrid-rag-pptx.pK3bDg/build_deck.mjs`
- Generate: `presentation/hybrid_rag_thesis_defense.pptx`

**Interfaces:**
- Consumes: the existing 28-slide data and `presentation/transcript.md`.
- Produces: a PPTX with matching palette, typography, diagram semantics, interleaved order and sourced speaker notes.

- [x] Add shared typography tokens and apply typeface explicitly to headings, body, labels and monospace evidence.
- [x] Keep core/detail backgrounds and semantic fills aligned with the HTML; remove any inconsistent dark treatment from supporting diagrams.
- [x] Add a small `RUNNING CASE · BaseRetriever.retrieve` treatment to the same slide groups used by the HTML.
- [x] Update relevant slide copy/notes to distinguish the representative trace from the 30-pair confirmatory workload and identify communities as a sibling global route.
- [x] Rebuild the final PPTX without changing benchmark numbers or RQ wording.

### Task 3: Cross-artifact visual and technical verification

**Files:**
- Inspect: `presentation/hybrid_rag_thesis_showcase.html`
- Inspect: `presentation/hybrid_rag_thesis_defense.pptx`
- Inspect: `/private/tmp/hybrid-rag-pptx.pK3bDg/rendered/`

**Interfaces:**
- Consumes: the completed HTML and PPTX from Tasks 1–2.
- Produces: verified render evidence and a clean working-tree report.

- [x] Render all 28 PPTX slides and inspect a full-deck montage plus representative full-size slides.
- [x] Run layout bounds checks and confirm zero overflow/clipping failures.
- [x] Confirm the PPTX imports and re-renders with 28 slides and that every slide retains speaker notes with `[Sources]`.
- [x] Inspect the HTML at default and narrow viewport sizes, including keyboard navigation and notes visibility.
- [x] Compare the two artifacts for palette, typography roles, running-case wording and semantic color consistency.
- [x] Report only the requested presentation artifacts and preserve unrelated worktree changes.
