from unittest.mock import MagicMock

from hybrid_rag.graph.community_builder import CommunityBuilder


def test_community_builder_separates_same_directory_across_repositories(monkeypatch):
    store = MagicMock()
    node_result = MagicMock()
    node_result.result_set = [
        ["repo-a::pkg.service", "service", "Module", "src/pkg/service.py", "repo-a"],
        ["repo-b::pkg.service", "service", "Module", "src/pkg/service.py", "repo-b"],
    ]
    edge_result = MagicMock()
    edge_result.result_set = []
    store.query.side_effect = [node_result, edge_result, *[MagicMock() for _ in range(5)]]
    monkeypatch.setattr(
        CommunityBuilder,
        "_generate_community_report",
        lambda _self, comm_id, _prompt: (comm_id, "summary"),
    )

    builder = CommunityBuilder(store)
    try:
        count = builder.build_communities()
    finally:
        builder.close()

    assert count == 2
