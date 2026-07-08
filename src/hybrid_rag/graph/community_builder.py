"""
Community Builder — partitions the codebase graph and compiles community summaries.

Uses directory structure partitioning for community detection, Ollama local model for
architectural summarization, and writes the resulting communities back to FalkorDB.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
import networkx as nx

from hybrid_rag.constants import DEFAULT_LLM_MODEL
from hybrid_rag.ports.graph_store import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = DEFAULT_LLM_MODEL
_HTTP_TIMEOUT = 120.0

_SYSTEM_PROMPT = (
    "You are an expert principal software architect. You write highly technical, structured, "
    "and clear architectural reports for codebase components. Use precise terminology."
)

_USER_TEMPLATE = """\
You are analyzing a specific functional community (cluster of files/classes/functions) of a codebase knowledge graph.

Analyze the community details below and write a comprehensive, high-quality architectural summary report.

Community Details:
- Community ID: {community_id}
- Directory Path: {dir_path}
- Number of Entities: {node_count}

Entity Members:
{entity_list}

Internal Relationships:
{relation_list}

Inter-Community Boundaries & Dependencies:
{boundary_list}

Requirements for your Report:
1. Title: Synthesize a highly descriptive, professional title (e.g., "Authentication and Session Management" or "Vector Store Index Adapters").
2. Primary Responsibility: Summarize the core purpose and architectural role of this community in 2-3 sentences.
3. Key Components: Detail the most critical classes, modules, or functions in this community and their responsibilities.
4. Usage Patterns & Data Flow: Describe how the elements within this community interact with each other and how other parts of the system interact with it.

Format your output EXACTLY like this:
TITLE: [Your synthesized title]
SUMMARY:
[Your detailed Markdown architectural report]
"""


class CommunityBuilder:
    """Partitions codebase graphs and generates community-level architectural summaries."""

    def __init__(
        self,
        graph_store: GraphStore,
        ollama_url: str | None = None,
        llm_model: str | None = None,
        timeout: float = _HTTP_TIMEOUT,
    ) -> None:
        self._store = graph_store
        self._ollama_url = (ollama_url or _DEFAULT_OLLAMA_URL).rstrip("/")
        self._llm_model = llm_model or _DEFAULT_MODEL
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def build_communities(self, resolution: float = 1.0) -> int:
        """
        Partition the graph, generate summaries via Ollama, and writeback to FalkorDB.

        Returns the number of communities created.
        """
        logger.info("Starting community building pipeline (resolution=%f)", resolution)

        # ── 1. Fetch entire graph from FalkorDB ──────────────────────────────
        nodes = self._fetch_all_nodes()
        edges = self._fetch_all_edges()

        if not nodes:
            logger.warning("FalkorDB graph is empty, skipping community detection.")
            return 0

        # Filter out existing Community nodes to avoid recursive partitioning
        code_nodes = [n for n in nodes if n["label"] != "Community"]
        code_node_ids = {n["id"] for n in code_nodes}
        code_edges = [
            e for e in edges if e["src_id"] in code_node_ids and e["dst_id"] in code_node_ids
        ]

        logger.info("Fetched %d code nodes and %d code edges", len(code_nodes), len(code_edges))

        if len(code_nodes) < 2:
            logger.warning("Not enough nodes for community detection.")
            # Create a single trivial community if there is at least one node
            self._write_trivial_community(code_nodes, code_edges)
            return 1

        # ── 2. Run Directory-Based Partitioning ──────────────────────────────
        G = nx.Graph()
        for node in code_nodes:
            G.add_node(node["id"], **node)
        for edge in code_edges:
            G.add_edge(edge["src_id"], edge["dst_id"], rel=edge["rel"])

        import os

        dir_to_nodes: dict[str, set[str]] = {}
        for node in code_nodes:
            fp = node.get("file_path", "")
            if fp:
                dir_path = os.path.dirname(fp)
                if not dir_path:
                    dir_path = "."
            else:
                dir_path = "root"
            dir_to_nodes.setdefault(dir_path, set()).add(node["id"])

        logger.info("Grouped codebase into %d directory-based communities", len(dir_to_nodes))

        # Map each code node ID to its community index
        node_to_comm: dict[str, int] = {}
        dir_paths = list(dir_to_nodes.keys())
        for comm_idx, dir_path in enumerate(dir_paths):
            for nid in dir_to_nodes[dir_path]:
                node_to_comm[nid] = comm_idx

        # ── 3. Compile and Summarize each Community ──────────────────────────
        compiled_communities = []

        for comm_idx, dir_path in enumerate(dir_paths):
            comm_id = f"community_lvl_0_{comm_idx}"
            comm_nodes = dir_to_nodes[dir_path]
            logger.info(
                "Summarizing community %d/%d (%s, Path: %s) with %d nodes…",
                comm_idx + 1,
                len(dir_paths),
                comm_id,
                dir_path,
                len(comm_nodes),
            )

            # Filter entities in this community
            comm_entities = [G.nodes[nid] for nid in comm_nodes]
            entity_list_str = "\n".join(
                f"  • {e['id']} (Type: {e['label']}, File: {e.get('file_path', '')})"
                for e in comm_entities
            )

            # Filter internal relations
            internal_edges = []
            boundary_edges = []
            for src_id in comm_nodes:
                # networkx neighbor iteration
                for dst_id in G.neighbors(src_id):
                    edge_data = G.get_edge_data(src_id, dst_id)
                    rel = edge_data.get("rel", "RELATED")
                    if dst_id in comm_nodes:
                        # Avoid double listing undirected edges
                        if src_id < dst_id:
                            internal_edges.append((src_id, rel, dst_id))
                    else:
                        boundary_edges.append((src_id, rel, dst_id, node_to_comm.get(dst_id, -1)))

            relation_list_str = (
                "\n".join(f"  • {src} -[:{rel}]-> {dst}" for src, rel, dst in internal_edges)
                if internal_edges
                else "  • No internal relationships."
            )

            boundary_list_str = (
                "\n".join(
                    f"  • {src} -[:{rel}]-> {dst} (Community #{target_idx})"
                    for src, rel, dst, target_idx in boundary_edges
                )
                if boundary_edges
                else "  • No external boundaries."
            )

            # Send to Ollama
            prompt = _USER_TEMPLATE.format(
                community_id=comm_id,
                dir_path=dir_path,
                node_count=len(comm_nodes),
                entity_list=entity_list_str,
                relation_list=relation_list_str,
                boundary_list=boundary_list_str,
            )

            title, summary = self._generate_community_report(comm_id, prompt)

            compiled_communities.append(
                {
                    "id": comm_id,
                    "name": title,
                    "summary": summary,
                    "nodes": list(comm_nodes),
                }
            )

        # ── 4. Write back to FalkorDB ─────────────────────────────────────────
        self._write_communities_to_db(compiled_communities, node_to_comm, code_edges)

        logger.info("Successfully wrote %d communities to FalkorDB", len(compiled_communities))
        return len(compiled_communities)

    def close(self) -> None:
        self._client.close()

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _fetch_all_nodes(self) -> list[dict[str, Any]]:
        """Fetch all nodes with plain Cypher projection to avoid driver object mismatches."""
        cypher = (
            "MATCH (n) RETURN n.id AS id, n.name AS name, labels(n)[0] AS label, "
            "n.file_path AS file_path, n.repository AS repository"
        )
        try:
            res = self._store.query(cypher)
            results = []
            for row in res.result_set or []:
                results.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "label": row[2],
                        "file_path": row[3],
                        "repository": row[4],
                    }
                )
            return results
        except Exception as exc:
            logger.error("Failed to fetch all nodes from FalkorDB: %s", exc)
            return []

    def _fetch_all_edges(self) -> list[dict[str, Any]]:
        """Fetch all edges with plain Cypher projection."""
        cypher = "MATCH (a)-[r]->(b) RETURN a.id AS src_id, type(r) AS rel, b.id AS dst_id"
        try:
            res = self._store.query(cypher)
            results = []
            for row in res.result_set or []:
                results.append(
                    {
                        "src_id": row[0],
                        "rel": row[1],
                        "dst_id": row[2],
                    }
                )
            return results
        except Exception as exc:
            logger.error("Failed to fetch all edges from FalkorDB: %s", exc)
            return []

    def _generate_community_report(self, comm_id: str, prompt: str) -> tuple[str, str]:
        """Call Ollama to generate the title and summary for a community."""
        payload = {
            "model": self._llm_model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 2048,
            },
        }
        try:
            resp = self._client.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()

            # Parse synthesized Title and Summary
            title = f"Community {comm_id}"
            summary = text

            title_match = re.search(r"^TITLE:\s*(.*)$", text, re.MULTILINE | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip().strip("[]\"'")

            summary_match = re.search(r"SUMMARY:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1).strip()

            # Clean up potential LLM markup leftovers
            if title.startswith("TITLE:"):
                title = title[6:].strip()

            return title, summary

        except Exception as exc:
            logger.warning("Ollama generation failed for %s: %s. Using fallback.", comm_id, exc)
            return (
                f"Community {comm_id} (Summary Generation Failed)",
                f"System was unable to contact Ollama model {self._llm_model} to generate report: {exc}",
            )

    def _write_trivial_community(
        self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Fallback for trivial or empty codebases."""
        comm_id = "community_lvl_0_0"
        node_ids = [n["id"] for n in nodes]
        entity_list_str = "\n".join(f"  • {n['id']} ({n['label']})" for n in nodes)
        summary = f"This single community aggregates all modules in a small codebase.\n\n### Entities:\n{entity_list_str}"

        self._store.query("MATCH (c:Community) DETACH DELETE c")
        self._store.query(
            "MERGE (c:Community {id: $id}) SET c.name = $name, c.summary = $summary, c.level = 0",
            {"id": comm_id, "name": "Codebase Core Core", "summary": summary},
        )
        if node_ids:
            self._store.query(
                "UNWIND $node_ids AS nid MATCH (n) WHERE n.id = nid "
                "MATCH (c:Community {id: $comm_id}) MERGE (n)-[:IN_COMMUNITY]->(c)",
                {"node_ids": node_ids, "comm_id": comm_id},
            )

    def _write_communities_to_db(
        self,
        communities: list[dict[str, Any]],
        node_to_comm: dict[str, int],
        edges: list[dict[str, Any]],
    ) -> None:
        """Clear previous communities and write new communities and links to FalkorDB."""
        # 1. Clear existing communities and edges
        self._store.query("MATCH (c:Community) DETACH DELETE c")

        # 2. Write new Community nodes
        for comm in communities:
            self._store.query(
                "MERGE (c:Community {id: $id}) SET c.name = $name, c.summary = $summary, c.level = 0",
                {"id": comm["id"], "name": comm["name"], "summary": comm["summary"]},
            )

            # 3. Create IN_COMMUNITY relationships for all member nodes in batch
            if comm["nodes"]:
                self._store.query(
                    "UNWIND $node_ids AS nid MATCH (n) WHERE n.id = nid "
                    "MATCH (c:Community {id: $comm_id}) MERGE (n)-[:IN_COMMUNITY]->(c)",
                    {"node_ids": list(comm["nodes"]), "comm_id": comm["id"]},
                )

        # 4. Write inter-community dependencies (COMMUNITY_DEPENDS)
        # Count connections between different communities
        comm_deps: dict[tuple[int, int], int] = {}
        for edge in edges:
            src_comm = node_to_comm.get(edge["src_id"])
            dst_comm = node_to_comm.get(edge["dst_id"])
            if src_comm is not None and dst_comm is not None and src_comm != dst_comm:
                key = (src_comm, dst_comm)
                comm_deps[key] = comm_deps.get(key, 0) + 1

        for (src_idx, dst_idx), weight in comm_deps.items():
            src_comm_id = f"community_lvl_0_{src_idx}"
            dst_comm_id = f"community_lvl_0_{dst_idx}"
            self._store.query(
                "MATCH (a:Community {id: $src}) MATCH (b:Community {id: $dst}) "
                "MERGE (a)-[r:COMMUNITY_DEPENDS]->(b) SET r.weight = $weight",
                {"src": src_comm_id, "dst": dst_comm_id, "weight": weight},
            )
