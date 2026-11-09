from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração por variáveis de ambiente (ou .env local, nunca versionado)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provedor de LLM compatível com a API de chat da OpenAI: Ollama, OmniRoute, Gemini (modo OpenAI).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3.5:4b"
    # JSON extra mesclado no corpo da requisição (ex.: desligar o raciocínio do Qwen).
    llm_extra_body: dict = {}
    llm_timeout_s: float = 60.0

    # Chave compartilhada com a API .NET (cabeçalho X-Api-Key). Vazia = sem checagem (só em dev).
    agent_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
