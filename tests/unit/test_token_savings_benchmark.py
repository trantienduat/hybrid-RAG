"""Repeatable token-footprint benchmark tests."""

import json

import pytest

from scripts.benchmark_token_savings import (
    calculate_comparison,
    main,
    scan_local_directory,
)


def test_scan_fails_when_no_files_match(tmp_path):
    (tmp_path / "README.md").write_text("hello")

    with pytest.raises(ValueError, match="zero readable files"):
        scan_local_directory(tmp_path, {".py"}, "gpt-4")


def test_scan_fails_on_unreadable_utf8_instead_of_skipping(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe")

    with pytest.raises(RuntimeError, match="unreadable benchmark files"):
        scan_local_directory(tmp_path, {".py"}, "gpt-4")


def test_calculation_labels_assumptions_and_never_claims_negative_savings():
    result = calculate_comparison(
        file_count=1,
        full_context_tokens=100,
        rag_prompt_tokens=200,
        hosted_cost_per_million=3.0,
        query_count=1000,
    )

    assert result["tokens_avoided_per_query"] == 0
    assert result["hypothetical_cost_avoided_usd"] == 0
    assert "assumptions" in result["evidence_class"]


def test_json_measurement_is_repeatable(tmp_path, capsys, monkeypatch):
    class FakeEncoding:
        name = "test_encoding"

        @staticmethod
        def encode(text):
            return text.split()

    monkeypatch.setattr("scripts.benchmark_token_savings._encoding", lambda _model: FakeEncoding())
    (tmp_path / "sample.py").write_text("def answer():\n    return 42\n")
    args = [str(tmp_path), "--rag-size", "5", "--queries", "10", "--json"]

    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["file_count"] == 1
    assert first["actual_encoding"]
