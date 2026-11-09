"""Interface de provedor de LLM (ADR 0004): trocar de modelo não mexe no resto do serviço."""

import re
from typing import Protocol

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(Exception):
    """Falha ao falar com o modelo (rede, status HTTP, resposta vazia)."""


class LLMProvider(Protocol):
    async def chat(
        self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.3
    ) -> str: ...


class OpenAICompatProvider:
    """Provedor para qualquer endpoint /chat/completions no formato OpenAI."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        extra_body: dict | None = None,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._extra_body = extra_body or {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    async def chat(
        self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        body: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            **self._extra_body,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.post("/chat/completions", json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"falha no provedor de LLM: {exc}") from exc
        # Modelos de raciocínio podem devolver o pensamento dentro da resposta.
        content = _THINK_RE.sub("", content or "").strip()
        if not content:
            raise LLMError("resposta vazia do modelo")
        return content

    async def aclose(self) -> None:
        await self._client.aclose()
