"""Lógica do agente: classificar risco, responder e atualizar a memória (ADR 0001, 0002)."""

import asyncio
import json
import re

from pydantic import ValidationError

from .provider import LLMError, LLMProvider
from .schemas import (
    Classificacao,
    ContextoAcompanhamento,
    MemoryUpdateRequest,
    RespondRequest,
    RespondResponse,
    Resposta,
)

MEMORIA_MAX_CHARS = 1500

CLASSIFICADOR_SISTEMA = """Você classifica a mensagem de um paciente de psicologia quanto ao nível de risco.
Responda APENAS um JSON: {"nivel": "nenhum" | "atencao" | "risco", "motivo": "<uma frase curta>"}

- "risco": qualquer sinal, direto ou velado, de pensar em morrer, sumir, desistir da vida, se machucar ou ser um peso para os outros; planos ou meios; desesperança grave.
- "atencao": sofrimento relevante sem sinal de risco: ansiedade forte, insônia persistente, choro, isolamento, conflito que abalou o paciente, perda de vontade de fazer as coisas.
- "nenhum": conversa do dia a dia, dúvida sobre tarefa ou exercício, relato neutro ou positivo, expressões idiomáticas sem sentido literal ("morrendo de rir", "esse trabalho me mata").

Regra de ouro: na dúvida entre dois níveis, escolha o MAIS ALTO. Um alarme falso é aceitável; um risco perdido não é.
Gatilhos definidos pelo psicólogo para este paciente (tratar pelo nível indicado): {gatilhos}
"""

RESPONDEDOR_SISTEMA = """Você é o agente de acompanhamento do PsiApoio. Você NÃO é terapeuta: é um companheiro entre as sessões, sob supervisão do psicólogo do paciente.

O que você PODE fazer:
- lembrar e acompanhar as tarefas combinadas com o psicólogo;
- tirar dúvidas sobre COMO executar um exercício cadastrado;
- perguntar e registrar como o paciente está se sentindo;
- acolher com calor humano.

O que você NUNCA faz: interpretar o que o paciente conta, diagnosticar, dar conselhos clínicos, indicar medicação, opinar sobre decisões de vida. Se o paciente pedir algo assim, acolha, diga que isso é importante e que você vai deixar registrado para ele conversar com o psicólogo na sessão, e marque fora_do_escopo=true com o tema em tema_para_pauta.
Você nunca promete sigilo: o psicólogo lê as conversas.

Tom: português do Brasil, natural, caloroso e direto, como uma pessoa atenciosa, não como um robô de atendimento. Respostas de 2 a 5 frases, sem listas, sem jargão técnico, sem emoji em excesso.

Perfil do paciente (aprovado pelo psicólogo):
{perfil}

Tarefas combinadas:
{tarefas}

Memória do acompanhamento:
{memoria}

Responda APENAS um JSON:
{{"resposta": "<texto ao paciente>", "fora_do_escopo": true|false, "tema_para_pauta": "<tema curto>" | null, "humor": <1 a 5 se o paciente indicou como está, senão null>}}
"""

MEMORIA_SISTEMA = """Você mantém a memória de um acompanhamento psicológico. Receba a memória atual e as mensagens novas e devolva a memória ATUALIZADA, em texto corrido e curto (no máximo {max} caracteres): fatos relevantes (rotina, preferências, eventos, tarefas feitas, humor recente, temas recorrentes). Sem interpretação clínica, sem diagnóstico, sem suposições. Mantenha o que continua válido e descarte o que ficou irrelevante. Responda APENAS o texto da memória."""


class AgentError(Exception):
    pass


def _extrair_json(texto: str) -> dict:
    """Aceita JSON puro ou JSON dentro de cercas de código / texto ao redor."""
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", texto, re.DOTALL)
        if not m:
            raise AgentError("a resposta do modelo não contém JSON")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise AgentError("JSON inválido na resposta do modelo") from exc


def _fmt_gatilhos(ctx: ContextoAcompanhamento) -> str:
    if not ctx.gatilhos:
        return "nenhum informado"
    return "; ".join(f'"{g.texto}" = {g.nivel}' for g in ctx.gatilhos)


async def classificar(llm: LLMProvider, mensagem: str, ctx: ContextoAcompanhamento) -> Classificacao:
    sistema = CLASSIFICADOR_SISTEMA.replace("{gatilhos}", _fmt_gatilhos(ctx))
    texto = await llm.chat(
        [{"role": "system", "content": sistema}, {"role": "user", "content": mensagem}],
        json_mode=True,
        temperature=0.0,
    )
    try:
        return Classificacao.model_validate(_extrair_json(texto))
    except ValidationError as exc:
        raise AgentError(f"classificação fora do formato: {exc}") from exc


async def responder(llm: LLMProvider, req: RespondRequest) -> Resposta:
    sistema = RESPONDEDOR_SISTEMA.format(
        perfil=req.perfil_agente or "(sem perfil)",
        tarefas="\n".join(f"- {t}" for t in req.tarefas) or "(nenhuma)",
        memoria=req.memoria or "(vazia)",
    )
    historico = [
        {"role": "user" if m.papel == "paciente" else "assistant", "content": m.texto}
        for m in req.mensagens_recentes
    ]
    texto = await llm.chat(
        [{"role": "system", "content": sistema}, *historico, {"role": "user", "content": req.mensagem}],
        json_mode=True,
        temperature=0.6,
    )
    try:
        return Resposta.model_validate(_extrair_json(texto))
    except ValidationError as exc:
        raise AgentError(f"resposta fora do formato: {exc}") from exc


async def processar_mensagem(llm: LLMProvider, req: RespondRequest) -> RespondResponse:
    """Classificação e resposta em paralelo (latência = a mais lenta das duas, não a soma).

    Falha de uma parte não derruba a outra: a API decide (ADR 0002). Risco None = não classificada.
    """
    classif, resp = await asyncio.gather(
        classificar(llm, req.mensagem, req),
        responder(llm, req),
        return_exceptions=True,
    )
    falhas = (LLMError, AgentError)
    risco = None if isinstance(classif, falhas) else classif
    resposta = None if isinstance(resp, falhas) else resp
    for r in (classif, resp):
        if isinstance(r, BaseException) and not isinstance(r, falhas):
            raise r  # bug de programação, não falha do modelo
    return RespondResponse(
        resposta=resposta.resposta if resposta else None,
        fora_do_escopo=resposta.fora_do_escopo if resposta else False,
        tema_para_pauta=resposta.tema_para_pauta if resposta else None,
        humor=resposta.humor if resposta else None,
        risco=risco.nivel if risco else None,
        motivo_risco=risco.motivo if risco else "",
    )


async def atualizar_memoria(llm: LLMProvider, req: MemoryUpdateRequest) -> str:
    conversa = "\n".join(f"{m.papel}: {m.texto}" for m in req.mensagens)
    texto = await llm.chat(
        [
            {"role": "system", "content": MEMORIA_SISTEMA.format(max=MEMORIA_MAX_CHARS)},
            {
                "role": "user",
                "content": f"Memória atual:\n{req.memoria or '(vazia)'}\n\nMensagens novas:\n{conversa}",
            },
        ],
        temperature=0.2,
    )
    return texto[:MEMORIA_MAX_CHARS]
