from data_junior_agent.config import AgentSettings


def test_settings_defaults_to_groq_model():
    settings = AgentSettings()
    assert settings.agent_provider == "litellm"
    assert settings.agent_model == "groq/openai/gpt-oss-120b"


def test_redacted_dict_hides_secrets():
    settings = AgentSettings(GROQ_API_KEY="gsk_secret_value")
    data = settings.redacted_dict()
    assert data["GROQ_API_KEY"] == "<set>"
    assert "secret" not in data["GROQ_API_KEY"]


def test_model_api_key_selects_groq_key():
    settings = AgentSettings(AGENT_MODEL="groq/openai/gpt-oss-120b", GROQ_API_KEY="gsk_test")
    assert settings.model_api_key() == "gsk_test"
