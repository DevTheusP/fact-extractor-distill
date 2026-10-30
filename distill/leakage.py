"""Checagem de vazamento: nenhuma mensagem de treino pode ser (quase) uma mensagem do benchmark."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

JACCARD_MAX = 0.35  # acima disso, a mensagem gerada é parecida demais com uma do benchmark

# Assuntos dos cenários do benchmark que o gerador não pode tocar.
BANNED_PATTERNS = [
    r"force[\s\-]*push", r"push\s+--force", r"verify\s*=\s*false", r"verifica[cç][aã]o\s+tls",
    r"payments?-sdk", r"servi[cç]o de pagamentos",
    # frases de abertura das mensagens do benchmark
    r"regra do time, importante", r"terminamos a migra[cç][aã]o", r"contexto do projeto:",
]


def _tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9][a-z0-9_.\-]*", t))


@lru_cache(maxsize=1)
def _bench_token_sets() -> list[tuple[str, set[str]]]:
    from .bench_data import bench_texts

    return [(t, _tokens(t)) for t in bench_texts()]


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def nearest_bench(text: str) -> tuple[float, str]:
    toks = _tokens(text)
    return max(((jaccard(toks, bt), t) for t, bt in _bench_token_sets()), key=lambda x: x[0])


def is_leak(text: str) -> bool:
    low = text.lower()
    if any(re.search(p, low) for p in BANNED_PATTERNS):
        return True
    return nearest_bench(text)[0] > JACCARD_MAX
