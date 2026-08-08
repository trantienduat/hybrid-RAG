# Thesis-Defense Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the existing Hybrid-RAG thesis showcase as an editorial evidence brief and upgrade its highest-value diagrams for formal defense readability.

**Architecture:** Keep one self-contained HTML deck. Modify the existing token system and component styles, replace selected CSS box flows with semantic inline SVG diagrams, and preserve the existing slide data model and keyboard navigation script.

**Tech Stack:** HTML, inline CSS, inline SVG, vanilla JavaScript, browser rendering, Python/Node read-only validation commands.

## Global Constraints

- Touch only `presentation/hybrid_rag_thesis_showcase.html` plus this design and plan documentation.
- Preserve all 15 core slides, 13 detail slides, presenter notes, hash navigation, keyboard controls, print behavior, and reduced-motion behavior.
- Keep all metric values, evidence boundaries, and implementation claims code-backed and unchanged unless a visual label needs clarification.
- Keep the artifact self-contained; do not add CDN fonts, JavaScript libraries, or external assets.

---

### Task 1: Apply the thesis-defense visual system

**Files:**
- Modify: `presentation/hybrid_rag_thesis_showcase.html:8-179`

**Interfaces:**
- Consumes: Existing `.slide`, `.depth`, `.note-box`, `.line-list`, `.metric-table`, and diagram component classes.
- Produces: CSS tokens and component styles used by every core and detail slide.

- [ ] Replace the neon-dark token values with the approved warm paper, ink navy, slate, teal, rust, and green evidence tokens.
- [ ] Set display headings to a local serif fallback stack and keep body copy in a system sans stack; reserve monospace for code and formulas.
- [ ] Reduce card radius and accent decoration; use borders, rules, whitespace, and asymmetrical columns to establish hierarchy.
- [ ] Add SVG semantic-shape classes, responsive text sizing, focus styling, reduced-motion rules, and print-safe token overrides.
- [ ] Preserve the existing mobile grid collapse and ensure the body has no accidental horizontal overflow.

### Task 2: Upgrade the diagrams that carry the thesis argument

**Files:**
- Modify: `presentation/hybrid_rag_thesis_showcase.html:240-339`
- Modify: `presentation/hybrid_rag_thesis_showcase.html:461-535`

**Interfaces:**
- Consumes: Existing slide claims and current routing/provenance/evaluation semantics.
- Produces: Inline SVG diagrams with stable `aria-labelledby` labels and CSS-controlled semantic roles.

- [ ] Replace the Core 05 index/route box flow with a two-stage SVG: one indexed source feeding graph and vector stores, followed by a decision diamond with exact-relation, generic-local, and global branches.
- [ ] Replace the Core 06 evidence map with two aligned SVG lanes: graph edges on one side and vector context on the other, both anchored to `BaseRetriever.retrieve`.
- [ ] Replace the Core 07 fusion block with graph/vector rank strips, an RRF operator, a context-selection stage, and visible route-weight annotations.
- [ ] Replace the Core 09 repeat/bootstrap flow with an SVG sequence that distinguishes raw records from the 30 paired statistical units.
- [ ] Replace the Core 08 detail provenance flow with explicit source, digest, index, preflight, and evaluation gates plus a visible abort branch.
- [ ] Upgrade the Core 05 global-community and Core 06 entity-resolution detail flows with clearer process/state semantics while keeping the existing wording.

### Task 3: Tighten defense hierarchy without changing claims

**Files:**
- Modify: `presentation/hybrid_rag_thesis_showcase.html` slide markup as needed

**Interfaces:**
- Consumes: Existing slide copy, source notes, and evidence labels.
- Produces: Claim/evidence/boundary grouping that remains compatible with the current navigation script.

- [ ] Add a consistent visual claim marker to core slides without introducing decorative numbering where sequence is not meaningful.
- [ ] Make uncertainty, rejected scope, and diagnostic-only evidence visually distinct from supported results.
- [ ] Keep the current metric tables and source notes, changing only labels or short bridge copy needed to clarify interpretation.
- [ ] Preserve all `data-axis`, `data-parent`, `data-depth`, and `data-depth-title` attributes so navigation remains unchanged.

### Task 4: Verify the deck before handoff

**Files:**
- Test: `presentation/hybrid_rag_thesis_showcase.html`

- [ ] Run `git diff --check` and inspect the diff to confirm unrelated dirty paths remain untouched.
- [ ] Run a structural HTML check for balanced sections, expected slide counts, navigation attributes, and required keyboard handlers.
- [ ] Run a JavaScript syntax check on the inline script and inspect the deck in a browser at wide and narrow viewports.
- [ ] Exercise core/depth navigation, transcript toggle, help, print mode, and reduced-motion behavior; fix any clipping, overlap, or illegible labels.
- [ ] Report the modified file, verification evidence, and any unverified browser limitation.
