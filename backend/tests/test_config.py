from retrievault.config import get_settings

def test_config_loads_defaults():
    # Make sure we don't accidentally load environment variables that override defaults
    # during this specific test.
    
    settings = get_settings()
    assert settings.corpus_repo == "fastapi/fastapi"
    assert settings.corpus_tag == "0.136.3"
    assert settings.retrievault_synthesis_model == "claude-sonnet-4-6"
