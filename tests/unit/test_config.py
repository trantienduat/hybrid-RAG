import os
import json
import tempfile
from pathlib import Path
from hybrid_rag.config import Config, DEFAULT_FALKORDB_HOST, DEFAULT_LLM_MODEL


def test_default_fallbacks():
    """Verify that when no config file is present, properties fallback to defaults or env."""
    config = Config(config_path="/nonexistent/config.json")
    
    # Check default fallback
    assert config.falkordb_host == DEFAULT_FALKORDB_HOST
    assert config.llm_model == DEFAULT_LLM_MODEL
    
    # Check env var override
    os.environ["FALKORDB_HOST"] = "test_env_host"
    os.environ["LLM_MODEL"] = "test_env_model"
    try:
        assert config.falkordb_host == "test_env_host"
        assert config.llm_model == "test_env_model"
    finally:
        del os.environ["FALKORDB_HOST"]
        del os.environ["LLM_MODEL"]


def test_config_file_loading():
    """Verify that values are parsed correctly from config.json."""
    sample_data = {
        "repositories": [
            {"name": "test-repo", "path": "/path/to/test-repo"}
        ],
        "falkordb": {
            "host": "config_host",
            "port": 9999
        },
        "llm": {
            "model": "config_llm_model"
        }
    }
    
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        json.dump(sample_data, tmp)
        tmp_path = tmp.name
        
    try:
        config = Config(config_path=tmp_path)
        
        # Check loaded file values
        assert config.falkordb_host == "config_host"
        assert config.falkordb_port == 9999
        assert config.llm_model == "config_llm_model"
        assert len(config.repositories) == 1
        assert config.repositories[0]["name"] == "test-repo"
        assert config.repositories[0]["path"] == "/path/to/test-repo"
        assert config.get_repo_path("test-repo") == "/path/to/test-repo"
        assert config.get_repo_path("nonexistent") is None
        
        # Verify env takes priority over config file
        os.environ["FALKORDB_HOST"] = "env_override_host"
        try:
            assert config.falkordb_host == "env_override_host"
        finally:
            del os.environ["FALKORDB_HOST"]
            
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
