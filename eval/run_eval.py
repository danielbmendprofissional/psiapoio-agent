"""Mede a classificação de risco de um modelo contra o conjunto rotulado (Q9 do discovery).

Uso (da raiz de psiapoio-agent):
    uv run python eval/run_eval.py --model qwen3.5:4b
    uv run python eval/run_eval.py --model qwen3.5:9b --repeat 3
    LLM_BASE_URL=... LLM_API_KEY=... uv run python eval/run_eval.py --model <modelo do OmniRoute>

O que importa é o RECALL DE RISCO (risco classificado como risco) e as subestimações:
um risco classificado como "nenhum" ou "atencao" é o erro grave.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psiapoio_agent import agent  # noqa: E402
from psiapoio_agent.provider import LLMError, OpenAICompatProvider  # noqa: E402
from psiapoio_agent.schemas import ContextoAcompanhamento  # noqa: E402
from psiapoio_agent.settings import get_settings  # noqa: E402

NIVEIS = ["nenhum", "atencao", "risco"]
ORDEM = {n: i for i, n in enumerate(NIVEIS)}
RAIZ = Path(__file__).resolve().parent


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=s.llm_model)
    ap.add_argument("--base-url", default=s.llm_base_url)
    ap.add_argument("--repeat", type=int, default=1, help="repete cada mensagem N vezes")
    args = ap.parse_args()

    dados = json.loads((RAIZ / "mensagens_rotuladas.json").read_text(encoding="utf-8"))
    llm = OpenAICompatProvider(
        args.base_url, s.llm_api_key, args.model, extra_body=s.llm_extra_body, timeout_s=s.llm_timeout_s
    )
    ctx = ContextoAcompanhamento(perfil_agente="")

    # Aquece o modelo (a primeira chamada paga o tempo de carga) para não contaminar a latência.
    try:
        await agent.classificar(llm, "oi", ctx)
    except (LLMError, agent.AgentError) as exc:
        print(f"[aviso] aquecimento falhou: {exc}")

    resultados = []
    for item in dados["mensagens"]:
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            try:
                c = await agent.classificar(llm, item["texto"], ctx)
                obtido, motivo, erro = c.nivel, c.motivo, None
            except (LLMError, agent.AgentError) as exc:
                obtido, motivo, erro = None, "", str(exc)
            resultados.append(
                {
                    "id": item["id"],
                    "texto": item["texto"],
                    "esperado": item["esperado"],
                    "obtido": obtido,
                    "motivo": motivo,
                    "erro": erro,
                    "latencia_s": round(time.perf_counter() - t0, 3),
                }
            )
    await llm.aclose()
    relatorio(args, resultados)


def relatorio(args, res: list[dict]) -> None:
    total = len(res)
    falhas = [r for r in res if r["obtido"] is None]
    validos = [r for r in res if r["obtido"] is not None]
    acertos = [r for r in validos if r["obtido"] == r["esperado"]]
    riscos = [r for r in res if r["esperado"] == "risco"]
    riscos_ok = [r for r in riscos if r["obtido"] == "risco"]
    subestimou = [r for r in validos if ORDEM[r["obtido"]] < ORDEM[r["esperado"]]]
    riscos_perdidos = [r for r in riscos if r["obtido"] != "risco"]  # inclui falha de parse
    superestimou = [r for r in validos if ORDEM[r["obtido"]] > ORDEM[r["esperado"]]]
    lat = sorted(r["latencia_s"] for r in validos) or [0.0]

    print(f"\nModelo: {args.model}   Base: {args.base_url}   Mensagens: {total}")
    print(f"Falhas de parse/rede: {len(falhas)}")
    print(f"Acurácia: {len(acertos)}/{len(validos)} = {len(acertos) / max(len(validos), 1):.0%}")
    print(
        f"RECALL DE RISCO: {len(riscos_ok)}/{len(riscos)} = {len(riscos_ok) / max(len(riscos), 1):.0%}"
        f"  (riscos perdidos: {len(riscos_perdidos)})"
    )
    print(f"Subestimou: {len(subestimou)}   Superestimou (alarme a mais): {len(superestimou)}")
    print(f"Latência por classificação: p50={statistics.median(lat):.2f}s  p95={lat[int(len(lat) * 0.95) - 1]:.2f}s")

    print("\nMatriz (linha = esperado, coluna = obtido):")
    cont = Counter((r["esperado"], r["obtido"]) for r in res)
    print(f"{'':10}" + "".join(f"{n:>9}" for n in NIVEIS) + f"{'falha':>9}")
    for e in NIVEIS:
        print(f"{e:10}" + "".join(f"{cont[(e, o)]:>9}" for o in NIVEIS) + f"{cont[(e, None)]:>9}")

    if riscos_perdidos or subestimou:
        print("\nERROS GRAVES (esperado > obtido):")
        for r in {r["id"]: r for r in riscos_perdidos + subestimou}.values():
            print(f"  #{r['id']} esperado={r['esperado']} obtido={r['obtido']}  «{r['texto']}»")

    pasta = RAIZ / "resultados"
    pasta.mkdir(exist_ok=True)
    nome = f"{args.model.replace(':', '_').replace('/', '_')}-{datetime.now():%Y%m%d-%H%M%S}.json"
    (pasta / nome).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalhes salvos em eval/resultados/{nome}")


if __name__ == "__main__":
    asyncio.run(main())
