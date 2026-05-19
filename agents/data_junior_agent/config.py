from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import AGENTS_ROOT, PROJECT_ROOT


class AgentSettings(BaseSettings):
    """Runtime settings loaded from environment variables or ignored .env files."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", AGENTS_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    agent_provider: str = Field(default="litellm", alias="AGENT_PROVIDER")
    agent_model: str = Field(default="groq/openai/gpt-oss-120b", alias="AGENT_MODEL")
    agent_max_turns: int = Field(default=10, alias="AGENT_MAX_TURNS", ge=1, le=50)
    agent_mcp_timeout_seconds: float = Field(default=30.0, alias="AGENT_MCP_TIMEOUT_SECONDS", ge=5.0, le=120.0)
    agent_trace: bool = Field(default=False, alias="AGENT_TRACE")
    agent_live_tests: bool = Field(default=False, alias="AGENT_LIVE_TESTS")
    agent_tool_max_chars: int = Field(default=2500, alias="AGENT_TOOL_MAX_CHARS", ge=500, le=12000)
    agent_max_scrapes: int = Field(default=2, alias="AGENT_MAX_SCRAPES", ge=0, le=10)
    agent_max_searches: int = Field(default=3, alias="AGENT_MAX_SEARCHES", ge=1, le=10)

    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    xai_api_key: SecretStr | None = Field(default=None, alias="XAI_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    firecrawl_api_key: SecretStr | None = Field(default=None, alias="FIRECRAWL_API_KEY")

    @field_validator("agent_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        provider = value.strip().lower()
        if provider != "litellm":
            raise ValueError("Only AGENT_PROVIDER=litellm is supported in this version.")
        return provider

    @field_validator("agent_model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("AGENT_MODEL cannot be empty.")
        return model

    def model_api_key(self) -> str | None:
        """Return the provider key that matches the configured LiteLLM model."""

        model = self.agent_model.lower()
        if model.startswith("groq/"):
            return self._secret_value(self.groq_api_key)
        if model.startswith("gemini/"):
            return self._secret_value(self.gemini_api_key)
        if model.startswith("xai/") or model.startswith("grok/"):
            return self._secret_value(self.xai_api_key)
        if model.startswith("openai/") or "/" not in model:
            return self._secret_value(self.openai_api_key)
        return None

    def require_model_api_key(self) -> str:
        api_key = self.model_api_key()
        if api_key:
            return api_key
        raise RuntimeError(
            f"No API key found for AGENT_MODEL={self.agent_model!r}. "
            "Set GROQ_API_KEY, GEMINI_API_KEY, XAI_API_KEY, or OPENAI_API_KEY as appropriate."
        )

    def firecrawl_key_value(self) -> str | None:
        return self._secret_value(self.firecrawl_api_key)

    def redacted_dict(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True, mode="json")
        for key in list(data):
            if key.endswith("_API_KEY"):
                field_name = key.lower()
                secret = getattr(self, field_name, None)
                data[key] = self._redact(self._secret_value(secret))
        return data

    @staticmethod
    def _secret_value(secret: SecretStr | None) -> str | None:
        if secret is None:
            return None
        value = secret.get_secret_value()
        return value or None

    @staticmethod
    def _redact(value: Any) -> str | None:
        if not value:
            return None
        return "<set>"


@lru_cache
def get_settings() -> AgentSettings:
    return AgentSettings()
