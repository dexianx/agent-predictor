import pytest

import task_a.inference.llm_client as llm


@pytest.mark.parametrize(
    ("provider", "model_env", "expected_label", "expected_model"),
    [
        ("gemini", "GEMINI_MODEL", "Gemini", "gemini-2.5-flash"),
        ("groq", "GROQ_MODEL", "Groq", "llama-3.3-70b-versatile"),
        ("cerebras", "CEREBRAS_MODEL", "Cerebras", "gpt-oss-120b"),
    ],
)
def test_llm_display_name_per_provider(monkeypatch, provider, model_env, expected_label, expected_model):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(model_env, expected_model)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(llm, "LLM_PROVIDER", provider, raising=False)
    assert expected_label in llm.llm_display_name()
    assert llm.active_model() == expected_model


def test_cerebras_openai_compat_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setattr(llm, "LLM_PROVIDER", "cerebras", raising=False)
    cfg = llm._openai_compat_config()
    assert cfg["base_url"] == "https://api.cerebras.ai/v1"
    assert cfg["api_key_env"] == "CEREBRAS_API_KEY"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "anthropic", raising=False)
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        llm._uses_openai_compat()
