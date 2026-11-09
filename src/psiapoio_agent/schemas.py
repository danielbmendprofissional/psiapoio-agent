from typing import Literal

from pydantic import BaseModel, Field

Nivel = Literal["nenhum", "atencao", "risco"]


class Gatilho(BaseModel):
    texto: str
    nivel: Literal["atencao", "risco"]


class MensagemHistorico(BaseModel):
    papel: Literal["paciente", "agente"]
    texto: str


class ContextoAcompanhamento(BaseModel):
    """Tudo o que o agente precisa vem na requisição: o serviço não tem estado nem banco (ADR 0003)."""

    perfil_agente: str = Field(description="Perfil aprovado pelo psicólogo (clínico + pessoal).")
    tarefas: list[str] = []
    gatilhos: list[Gatilho] = []
    memoria: str = ""
    mensagens_recentes: list[MensagemHistorico] = []


class RespondRequest(ContextoAcompanhamento):
    mensagem: str


class Resposta(BaseModel):
    resposta: str
    fora_do_escopo: bool = False
    tema_para_pauta: str | None = None
    humor: int | None = Field(default=None, ge=1, le=5)


class Classificacao(BaseModel):
    nivel: Nivel
    motivo: str = ""


class RespondResponse(BaseModel):
    """Qualquer campo None significa falha daquela parte: a API decide o que fazer (ADR 0002).

    `risco` None = mensagem não classificada, nunca tratar como "nenhum".
    """

    resposta: str | None
    fora_do_escopo: bool = False
    tema_para_pauta: str | None = None
    humor: int | None = None
    risco: Nivel | None
    motivo_risco: str = ""


class MemoryUpdateRequest(BaseModel):
    memoria: str = ""
    mensagens: list[MensagemHistorico]


class MemoryUpdateResponse(BaseModel):
    memoria: str
