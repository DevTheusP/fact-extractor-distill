"""Conjunto de teste vindo do agent-memory-bench (nunca entra no treino).

Duas fontes, ambas copiadas do repositório original (MIT, mesmo autor):

- resources/bench/messages.json: as 12 mensagens dos 4 cenários e as 24 frases
  de ruído de scenarios/library.py, com o gabarito de cada cenário.
- resources/bench/extract_cache_ollama_qwen2.5_7b.json: o cache de extração da
  rodada publicada (qwen2.5:7b via Ollama). Cada entrada é um par exato
  (mensagem, chaves existentes) que o benchmark pediu ao extrator, com a
  resposta do professor original. Usamos os pares como entradas de teste: são
  exatamente as chamadas que o aluno vai receber quando substituir o professor.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .generate import Example, Intent

RESOURCES = Path(__file__).parent / "resources" / "bench"


@lru_cache(maxsize=1)
def bench_messages() -> dict:
    return json.loads((RESOURCES / "messages.json").read_text(encoding="utf-8"))


def bench_texts() -> list[str]:
    data = bench_messages()
    return [m["text"] for m in data["core"] + data["noise"]]


def ollama_cache() -> list[tuple[str, list[str], list[dict]]]:
    raw = json.loads((RESOURCES / "extract_cache_ollama_qwen2.5_7b.json").read_text(encoding="utf-8"))
    return [(k[0], list(k[1]), facts) for k, facts in raw]


# Chave existente que descreve o mesmo atributo do gabarito (o professor escolhe
# o nome; ex.: payment_service.region). Se existir, o certo é reutilizá-la.
_EQUIVALENT = {
    "payments.aws_region": r"^payment[a-z_]*\.[a-z_]*region",
    "backend.test_framework": r"^backend\.test",
    "frontend.test_framework": r"^frontend\.test",
    "git.main.force_push": r"force_push",
}


def _existing_equivalent(gold_key: str, existing: list[str]) -> str | None:
    pat = _EQUIVALENT.get(gold_key)
    return next((k for k in existing if pat and re.search(pat, k)), None)


def bench_test_cases() -> list[Example]:
    """Os pares (mensagem, chaves) do benchmark como exemplos de teste.

    A intenção vem do gabarito do cenário quando a mensagem afirma um fato
    (Fact do benchmark); reforços e ruído ficam sem intenção.
    """
    by_text = {m["text"]: m for m in bench_messages()["core"] + bench_messages()["noise"]}
    out = []
    for i, (text, keys, _) in enumerate(ollama_cache()):
        m = by_text.get(text)
        intents = []
        category = "bench_noise"
        if m is not None and m.get("scenario"):
            category = f"bench_{m['scenario']}"
            gold = m.get("gold")
            if gold and gold.get("trusted", True):
                kind = "procedural" if m["scenario"] == "rare_fact" else "semantic"
                key = _existing_equivalent(gold["subject_key"], keys) or gold["subject_key"]
                intents = [Intent(key, gold["value"], kind)]
            elif m.get("reinforces"):
                # reforço (ex.: "Subi o hotfix ... em us-east-1"): extrair de novo é opcional
                region = re.search(r"\b[a-z]{2}-[a-z]+-\d\b", text)
                intents = [Intent(m["reinforces"], region.group(0) if region else "", "semantic", optional=True)]
        reuse = next((it.key for it in intents if not it.optional and it.key in keys), None)
        out.append(Example(id=f"bench-{i:03d}", text=text, existing_keys=keys, category=category,
                           source=(m or {}).get("source", "user"), intents=intents, reuse_key=reuse,
                           split="test_bench"))
    return out


def ollama_labels() -> dict[tuple[str, tuple[str, ...]], list[dict]]:
    """Rótulos do professor original (Ollama, q4) por (texto, chaves)."""
    return {(t, tuple(k)): facts for t, k, facts in ollama_cache()}
