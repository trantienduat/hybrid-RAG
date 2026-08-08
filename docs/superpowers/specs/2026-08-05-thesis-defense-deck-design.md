# Thesis-Defense Deck Design

**Date:** 2026-08-05  
**Artifact:** `presentation/hybrid_rag_thesis_showcase.html`

## Goal

Reframe the existing Hybrid-RAG showcase as a formal thesis-defense deck: make claims, evidence, and limits easier to scan while preserving the existing code-backed wording, presenter notes, keyboard navigation, supporting-detail slides, and print behavior.

## Visual direction

- Use a deliberate editorial evidence-brief register: warm paper ground, ink navy headings, slate body text, muted teal evidence, and rust boundaries or warnings.
- Use a serif display face for thesis claims, a system sans face for explanatory copy, and monospace only for code, paths, formulas, and metrics.
- Replace repeated high-radius card styling with ruled sections, restrained borders, asymmetric columns, and stronger whitespace.
- Keep the artifact single-theme by intent, but express all colors through CSS tokens so inline SVG diagrams inherit the theme and print legibly.

## Diagram direction

Use the diagram grammar that matches each question rather than one universal box-and-arrow pattern:

- Query routing: decision flow with three explicit paths: exact relation, generic local, and global.
- Evidence representation: two coordinated lanes showing structural graph facts versus semantic vector context.
- RRF: aligned graph/vector rank strips converging into context selection, with route-dependent weights visible.
- Provenance and evaluation: sequential gates and transformations with explicit stop/uncertain branches.
- Demo: request trace from repository-scoped API call through retrieval, local generation, and cited sources.

Use inline SVG for relational/process diagrams, semantic shapes for decisions and stores, orthogonal connectors with readable labels, and optional path highlighting that does not hide the static overview. Do not add external libraries or alter the deck's existing navigation model.

## Content and evidence constraints

- Do not invent metrics, benchmark results, or implementation capabilities.
- Preserve the current distinction between measured evidence, diagnostic evidence, uncertainty, and future work.
- The architecture diagram must reflect current routing behavior: exact relation can return graph-only, generic local fuses graph and vector, and global retrieval uses directory-based community summaries.
- Keep existing transcript sources and update only wording required by the visual hierarchy.

## Acceptance criteria

- The same 15 core slides and 13 supporting-detail slides remain reachable by the current keyboard controls.
- Core claim slides have a clear visual hierarchy and readable projector-scale text.
- High-value diagrams have visually distinct semantic roles and no connector/label collisions at the default viewport.
- `N`/`T` transcript toggle, `H` help, deep-link hashes, print layout, reduced-motion behavior, and mobile fallback remain functional.
- The HTML remains self-contained with inline CSS and JavaScript.
- Fresh structural, syntax, whitespace, and rendered visual checks are run before claiming completion.
