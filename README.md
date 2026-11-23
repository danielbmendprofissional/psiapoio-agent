# psiapoio-agent

Serviço **Python (FastAPI)** do agente do **PsiApoio**. É **sem estado e sem acesso ao banco**: a API .NET manda todo o contexto a cada chamada e este serviço devolve resposta, nível de risco e memória atualizada. Só a API chama este serviço, com a chave `X-Api-Key` (ADR 0003). Parte do projeto [PsiApoio](https://github.com/danielbmendprofissional/psiapoio).

| Repositório | Papel |
|---|---|
| [psiapoio](https://github.com/danielbmendprofissional/psiapoio) | Vitrine e visão geral |
| [psiapoio-workspace](https://github.com/danielbmendprofissional/psiapoio-workspace) | Specs, ADRs, glossário |
| [psiapoio-api](https://github.com/danielbmendprofissional/psiapoio-api) | API .NET |
| [psiapoio-frontend](https://github.com/danielbmendprofissional/psiapoio-frontend) | Front Angular |
| **psiapoio-agent** (este) | Serviço do agente |

## O que faz

- **Escopo fechado (ADR 0001):** não é terapeuta. Reforça tarefas combinadas, tira dúvida de como executar um exercício, registra humor e acolhe. Qualquer outra coisa vira resposta de acolhimento + `fora_do_escopo` + tema para a Pauta.
- **Risco por mensagem (ADR 0002):** classifica em `nenhum`, `atencao` ou `risco` (na dúvida, o mais alto). Classificação e resposta rodam **em paralelo**. Em `risco`, **quem usa a mensagem fixa é a API**; a resposta do LLM é descartada. `risco: null` significa "não classificada" e nunca deve ser tratada como "nenhum".
- **Memória:** resumo corrido de até 1500 caracteres, atualizado a pedido da API.
- **Resumo diário e Pauta:** anotações sem interpretação clínica.
- **LLM atrás de uma interface (ADR 0004):** qualquer provedor compatível com a API de chat da OpenAI (Ollama, OmniRoute, Gemini em modo OpenAI), trocando só variáveis de ambiente.

## Rodar

Pré-requisitos: Python 3.12+ via [`uv`](https://docs.astral.sh/uv/) e um provedor de LLM (ex.: Ollama com `qwen3.5:9b`).

```bash
cp .env.example .env                 # ajuste LLM_MODEL e AGENT_API_KEY
uv run uvicorn psiapoio_agent.main:app --app-dir src --port 8000
```

Variáveis (`.env`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_EXTRA_BODY`, `LLM_TIMEOUT_S`, `AGENT_API_KEY` (a mesma `Agent:ApiKey` da API; vazia = sem checagem, só em dev).

> A primeira chamada ao modelo local costuma ser lenta. Aqueça-o antes de uma demonstração.

## Endpoints

Todos (exceto `/health`) exigem o cabeçalho `X-Api-Key`.

| Rota | O que faz |
|---|---|
| `GET /health` | liveness |
| `POST /v1/respond` | contexto + mensagem → `resposta`, `fora_do_escopo`, `tema_para_pauta`, `humor`, `risco`, `motivo_risco`. Campo `null` = aquela parte falhou |
| `POST /v1/memory/update` | memória + mensagens novas → memória atualizada |
| `POST /v1/summary` | **Resumo diário** de um dia |
| `POST /v1/agenda` | **Pauta** da próxima sessão a partir de resumos e temas |

O contrato (formatos de entrada e saída) está em `src/psiapoio_agent/schemas.py` e descrito no workspace, em `specs/transversal.md`, seção 3.

## Testes

```bash
uv run pytest
```

15 testes com provedor falso, sem rede: paralelismo, falhas parciais, JSON inválido, gatilhos no prompt, autenticação por chave, resumo e pauta.

## Avaliação da classificação de risco

`eval/mensagens_rotuladas.json` tem 32 mensagens **fictícias** rotuladas. O relatório mostra acurácia, matriz de confusão, latência e, principalmente, o **recall de risco** e os erros graves (risco classificado abaixo do real).

```bash
uv run python eval/run_eval.py --model qwen3.5:9b
```

Última medição (2 passadas das 32 mensagens, 2026-11-09): recall de risco **22/22** em `qwen3.5:4b` e `qwen3.5:9b`; acurácia geral 60/64 e 58/64. A detecção por LLM **não é infalível**; essa limitação é declarada no ADR 0002.

## Aviso

Na demonstração só entram dados fictícios: os tiers gratuitos de LLM podem usar o conteúdo para treino (ADR 0004). Não use com pacientes reais.
