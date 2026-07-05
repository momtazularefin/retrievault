from retrievault.config import Settings


def test_config_loads_defaults():
    # Make sure we don't accidentally load environment variables that override defaults
    # during this specific test.
    settings = Settings(_env_file=None)

    assert settings.corpus_repo == "fastapi/fastapi"
    assert settings.corpus_tag == "0.136.3"
    assert settings.retrievault_synthesis_model == "claude-sonnet-4-6"
    assert settings.qdrant_api_key_or_none is None


def test_config_preserves_non_empty_qdrant_api_key():
    settings = Settings(_env_file=None, qdrant_api_key=" secret ")

    assert settings.qdrant_api_key_or_none == "secret"
