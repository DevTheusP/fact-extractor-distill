"""Leitura da saída do modelo: JSON válido, conforme o schema, e normalização dos fatos."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

from .prompt import SCHEMA

_VALIDATOR = Draft202012Validator(SCHEMA)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class Fact:
    subject_key: str
    value: str
    kind: str
    statement: str

    def to_dict(self) -> dict:
        return {"subject_key": self.subject_key, "value": self.value,
                "kind": self.kind, "statement": self.statement}


@dataclass
class Parsed:
    """Resultado da leitura. facts só é confiável quando schema_ok é verdadeiro."""

    json_ok: bool
    schema_ok: bool
    facts: list[Fact] = field(default_factory=list)
    error: str | None = None


def parse_output(text: str) -> Parsed:
    """Interpreta a resposta crua do modelo.

    Aceita cercas de markdown (```json ... ```), comuns em modelos pequenos sem
    gramática, mas conta como inválido qualquer outro texto em volta do JSON:
    o LLMExtractor do benchmark faz json.loads direto na resposta.
    """
    raw = text.strip()
    m = _FENCE.match(raw)
    if m:
        raw = m.group(1)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return Parsed(False, False, error=f"json: {e}")
    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        return Parsed(True, False, facts=_lenient_facts(data), error=f"schema: {errors[0].message}")
    facts = [Fact(f["subject_key"].strip(), f["value"].strip(), f["kind"], f["statement"].strip())
             for f in data["facts"] if f["subject_key"].strip() and f["value"].strip()]
    return Parsed(True, True, facts=facts)


def _lenient_facts(data) -> list[Fact]:
    """Aproveita o que der de um JSON fora do schema (só para inspeção, não para métrica)."""
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        return []
    out = []
    for f in data["facts"]:
        if isinstance(f, dict) and isinstance(f.get("subject_key"), str) and isinstance(f.get("value"), str):
            out.append(Fact(f["subject_key"].strip(), f["value"].strip(), str(f.get("kind", "")),
                            str(f.get("statement", ""))))
    return out


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def normalize_key(key: str) -> str:
    """billing.Database-Engine -> billing.database_engine."""
    k = _strip_accents(key).strip().lower()
    k = re.sub(r"[\s\-]+", "_", k)
    k = re.sub(r"_*\._*", ".", k)
    return k.strip("._")


def normalize_value(value: str) -> str:
    """Compara valores sem caixa, acento, aspas, crase e pontuação final."""
    v = _strip_accents(value).strip().lower()
    v = re.sub(r"\s+", " ", v).rstrip(".;,! ")
    return v.strip("`'\"“”‘’ ")


def to_json(facts: list[Fact] | list[dict]) -> str:
    """Serialização canônica do alvo de treino: uma linha, UTF-8 legível, campos na ordem do schema."""
    items = [f.to_dict() if isinstance(f, Fact) else
             {k: f[k] for k in ("subject_key", "value", "kind", "statement")} for f in facts]
    return json.dumps({"facts": items}, ensure_ascii=False)
