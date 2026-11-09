import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from . import agent
from .provider import LLMError, LLMProvider, OpenAICompatProvider
from .schemas import (
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    RespondRequest,
    RespondResponse,
)
from .settings import Settings, get_settings

log = logging.getLogger("psiapoio_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if not s.agent_api_key:
        log.warning("AGENT_API_KEY vazia: chamadas sem autenticação (só aceitável em dev)")
    app.state.llm = OpenAICompatProvider(
        s.llm_base_url,
        s.llm_api_key,
        s.llm_model,
        extra_body=s.llm_extra_body,
        timeout_s=s.llm_timeout_s,
    )
    yield
    await app.state.llm.aclose()


app = FastAPI(title="PsiApoio Agent", lifespan=lifespan)


def exigir_chave(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Só a API .NET chama este serviço (ADR 0003), com a chave compartilhada."""
    if settings.agent_api_key and not secrets.compare_digest(
        x_api_key or "", settings.agent_api_key
    ):
        raise HTTPException(status_code=401, detail="chave inválida")


def get_llm() -> LLMProvider:
    return app.state.llm


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/respond", response_model=RespondResponse, dependencies=[Depends(exigir_chave)])
async def respond(req: RespondRequest, llm: LLMProvider = Depends(get_llm)) -> RespondResponse:
    return await agent.processar_mensagem(llm, req)


@app.post(
    "/v1/memory/update",
    response_model=MemoryUpdateResponse,
    dependencies=[Depends(exigir_chave)],
)
async def memory_update(
    req: MemoryUpdateRequest, llm: LLMProvider = Depends(get_llm)
) -> MemoryUpdateResponse:
    try:
        return MemoryUpdateResponse(memoria=await agent.atualizar_memoria(llm, req))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="falha no modelo") from exc
