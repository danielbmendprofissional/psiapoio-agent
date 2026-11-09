import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from psiapoio_agent import agent
from psiapoio_agent.main import app, get_llm
from psiapoio_agent.provider import LLMError
from psiapoio_agent.schemas import Gatilho, RespondRequest
from psiapoio_agent.settings import Settings, get_settings


class FakeLLM:
    """Responde conforme o prompt de sistema: classificador ou respondedor."""

    def __init__(self, nivel="nenhum", falha_classif=False, falha_resp=False, classif_raw=None):
        self.nivel = nivel
        self.falha_classif = falha_classif
        self.falha_resp = falha_resp
        self.classif_raw = classif_raw
        self.chamadas: list[list[dict]] = []

    async def chat(self, messages, *, json_mode=False, temperature=0.3):
        self.chamadas.append(messages)
        sistema = messages[0]["content"]
        if "classifica a mensagem" in sistema:
            if self.falha_classif:
                raise LLMError("fora do ar")
            if self.classif_raw is not None:
                return self.classif_raw
            return json.dumps({"nivel": self.nivel, "motivo": "teste"})
        if "agente de acompanhamento" in sistema:
            if self.falha_resp:
                raise LLMError("fora do ar")
            return json.dumps(
                {"resposta": "Que bom que você fez o exercício.", "fora_do_escopo": False,
                 "tema_para_pauta": None, "humor": 4}
            )
        return "memória nova"


def req(msg="Fiz o exercício hoje"):
    return RespondRequest(perfil_agente="Prefere tom leve.", mensagem=msg, tarefas=["Respiração 4-7-8"])


async def test_classificacao_e_resposta_em_paralelo():
    llm = FakeLLM(nivel="atencao")
    out = await agent.processar_mensagem(llm, req())
    assert out.risco == "atencao"
    assert out.resposta == "Que bom que você fez o exercício."
    assert out.humor == 4
    assert len(llm.chamadas) == 2


async def test_falha_do_classificador_deixa_risco_nulo_e_mantem_resposta():
    out = await agent.processar_mensagem(FakeLLM(falha_classif=True), req())
    assert out.risco is None  # nunca vira "nenhum"
    assert out.resposta is not None


async def test_json_invalido_do_classificador_deixa_risco_nulo():
    out = await agent.processar_mensagem(FakeLLM(classif_raw="não sei dizer"), req())
    assert out.risco is None


async def test_nivel_desconhecido_deixa_risco_nulo():
    out = await agent.processar_mensagem(FakeLLM(classif_raw='{"nivel": "talvez"}'), req())
    assert out.risco is None


async def test_falha_da_resposta_mantem_classificacao():
    out = await agent.processar_mensagem(FakeLLM(nivel="risco", falha_resp=True), req())
    assert out.risco == "risco"
    assert out.resposta is None


async def test_json_dentro_de_cerca_de_codigo():
    raw = '```json\n{"nivel": "risco", "motivo": "x"}\n```'
    out = await agent.processar_mensagem(FakeLLM(classif_raw=raw), req())
    assert out.risco == "risco"


async def test_requisicoes_rodam_em_paralelo():
    class Lento(FakeLLM):
        async def chat(self, messages, **kw):
            await asyncio.sleep(0.2)
            return await super().chat(messages, **kw)

    inicio = asyncio.get_event_loop().time()
    await agent.processar_mensagem(Lento(), req())
    assert asyncio.get_event_loop().time() - inicio < 0.35  # ~0,2s, não ~0,4s


async def test_gatilhos_entram_no_prompt_do_classificador():
    llm = FakeLLM()
    r = req()
    r.gatilhos = [Gatilho(texto="prova final", nivel="atencao")]
    await agent.processar_mensagem(llm, r)
    classif = next(c for c in llm.chamadas if "classifica a mensagem" in c[0]["content"])
    assert "prova final" in classif[0]["content"]


@pytest.fixture
def client():
    llm = FakeLLM(nivel="risco")
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_settings] = lambda: Settings(agent_api_key="segredo")
    yield TestClient(app)
    app.dependency_overrides.clear()


BODY = {"perfil_agente": "x", "mensagem": "oi"}


def test_api_exige_chave(client):
    assert client.post("/v1/respond", json=BODY).status_code == 401
    assert client.post("/v1/respond", json=BODY, headers={"X-Api-Key": "errada"}).status_code == 401


def test_api_com_chave_correta(client):
    r = client.post("/v1/respond", json=BODY, headers={"X-Api-Key": "segredo"})
    assert r.status_code == 200
    assert r.json()["risco"] == "risco"


def test_health_sem_chave(client):
    assert client.get("/health").status_code == 200
