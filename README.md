# psiapoio-agent

Serviço Python do agente do PsiApoio. **Sem estado e sem acesso ao banco** (ADR 0003): a API .NET manda o contexto em cada chamada e este serviço devolve resposta, nível de risco e memória atualizada. Só a API chama este serviço, com a chave `X-Api-Key`.

## Rodar

```bash
cp .env.example .env        # ajuste o provedor de LLM
uv run uvicorn psiapoio_agent.main:app --reload --app-dir src
uv run pytest               # testes (provedor falso, sem rede)
```

## Endpoints

| Rota | O que faz |
|---|---|
| `GET /health` | liveness |
| `POST /v1/respond` | recebe contexto + mensagem; classificação de risco e resposta rodam **em paralelo**. `risco: null` = não classificada (nunca tratar como "nenhum"); `resposta: null` = falhou, a API usa a mensagem padrão. Em `risco`, **quem usa a mensagem fixa é a API** (ADR 0002) |
| `POST /v1/memory/update` | recebe memória + mensagens novas e devolve a memória atualizada (a API chama em segundo plano, depois da resposta) |

## Provedor de LLM

`provider.py` define a interface; `OpenAICompatProvider` serve Ollama, OmniRoute e Gemini (modo OpenAI) trocando só variáveis de ambiente (`.env.example`). **Na demo, só dados fictícios** (ADR 0004).

## Medir a classificação de risco (Q9)

`eval/mensagens_rotuladas.json` tem 32 mensagens fictícias rotuladas. **Revise os rótulos**: são uma primeira proposta.

```bash
uv run python eval/run_eval.py --model qwen3.5:4b
uv run python eval/run_eval.py --model qwen3.5:9b
```

O relatório mostra acurácia, matriz de confusão, latência (p50/p95) e, principalmente, o **recall de risco** e os erros graves (risco classificado abaixo do real).

## Fora do escopo desta fatia

Streaming da resposta (a API vai reter a resposta até a classificação chegar; fica para a fatia do chat), e o formulário/perfil do agente (a API monta `perfil_agente`).
