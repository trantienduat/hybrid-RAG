"""Local-only CLI serving defaults."""

from unittest.mock import patch

from typer.testing import CliRunner

from hybrid_rag.cli import app


def test_serve_defaults_to_loopback_and_one_worker():
    runner = CliRunner()

    with patch("uvicorn.run") as run:
        result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["workers"] == 1
