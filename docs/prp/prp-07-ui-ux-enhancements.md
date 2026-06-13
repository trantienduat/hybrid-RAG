# Product Requirement Prompt (PRP) — UI/UX Overhaul & Advanced Chat Features

## 🎯 Role & Objective
You are a senior frontend developer and UI/UX designer. Your task is to implement advanced UI/UX features, persistent chat history, and streaming controllers for the `hybrid-RAG` platform. The interface must transition into a premium **Codebase Architect Explorer** featuring split-pane resizing, local-storage conversation management, active LLM stream aborting, markdown rendering, and collapsible reasoning tracks.

---

## 🏛️ Tech Stack & Library Integrations
*   **Markdown Parsing:** `marked.js` loaded via CDN.
*   **Aesthetics:** Premium cyber-dark mode, glassmorphism containers, Outfit / Inter Google typography, and accent colors (cyan, purple, and warning red).
*   **State Persistence:** Browser `localStorage`.

---

## 🛠️ Functional Requirements

### 1. Drag-to-Resize Layout Panels
*   Add vertical divider bars (`.resizer`) between:
    *   Left Sidebar (Search & History)
    *   Center Pane (Graph Canvas visualizer)
    *   Right Pane (Q&A Chat)
*   Implement drag-to-resize JavaScript handlers with strict boundaries (min-width / max-width limit constraints).
*   Trigger Cytoscape (`cyInstance.resize()`) and 3D Force-Graph canvas size adjustments reactively on resizer drag events to avoid visual clipping.

### 2. LocalStorage Conversation History
*   Store all chats in `localStorage` under `hybrid_rag_chats`.
*   Auto-generate conversation titles using the first user prompt text.
*   Provide a history list in the Left Sidebar. Clicking a conversation loads its messages, repository select, and codebase RAG toggle state.
*   Display a hover-delete button (✕) to delete a conversation.
*   **Safe ID Collision Fix**: When loading a conversation, update the global message ID counter `msgIdCounter` to the maximum ID of the loaded messages. Reset `msgIdCounter` to `0` when starting a new chat or deleting/clearing the current session.

### 3. Active Stream Interruption (Stop Button)
*   Bind stream requests to an `AbortController`.
*   During active token generation, change the "Ask" button to a red "Stop" button (`.btn-stop`).
*   Clicking the Stop button aborts the HTTP request, halts the stream, and logs a *"Generation stopped by user"* message.
*   Hook up automatic aborts when switching conversations, deleting chats, or starting a new session.

### 4. Collapsible LLM Reasoning (Thinking Box)
*   Extract step-by-step model reasoning wrapped inside `<think>...</think>` tags.
*   Render the extracted thoughts inside a collapsible `<details class="thinking-box">` element styled with glassmorphism.
*   Render the remaining response as markdown below the thinking details box.

### 5. RAG Trace & Suggestion Chips
*   Render badge indicators for Query Classification (HYBRID, VECTOR, GRAPH, GENERAL), retrieved source count, and Reciprocal Rank Fusion (RRF) metrics.
*   Display search suggestion chips and recent query history dynamically under the search bar.

---

## 📈 Non-Functional Requirements & Performance
*   **No Stream Truncation:** Ensure client-side AbortController matches backend-side timeouts (HTTPX timeouts set to `300s` to support long codebase prompt evaluations).
*   **Clean Rendering:** Code blocks, lists, and tables inside Markdown responses must align nicely with the dark-theme layouts.
