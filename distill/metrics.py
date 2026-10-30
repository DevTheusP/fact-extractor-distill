"""Métricas do extrator: aluno (ou professor) contra uma referência.

Por caso, compara a lista prevista com a de referência:

- json_ok / schema_ok: a resposta crua é JSON e segue o SCHEMA do benchmark.
  Resposta fora do schema conta como lista vazia nas demais métricas.
- fato estrito: subject_key normalizada igual E value normalizado igual
  (pareamento um para um). É o que importa para a memória: chave diferente
  vira fato novo em vez de substituir o antigo.
- fato por valor: só o value (igual ou um contido no outro). Mede se o fato
  certo foi extraído mesmo quando o nome da chave diverge.
- kind: acerto de semantic/procedural nos pares casados por valor.
- reutilização: quando a referência reutiliza uma chave existente, o previsto
  usa exatamente a mesma string? (o store do benchmark busca a chave exata)
- reutilização indevida: o previsto usa uma chave existente que a referência
  não usou.
- lista vazia: quando a referência é vazia, o previsto também é?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean, median

from .parsing import Fact, Parsed, normalize_key, normalize_value


def _as_fact(f) -> Fact:
    if isinstance(f, Fact):
        return f
    return Fact(f["subject_key"].strip(), f["value"].strip(), f.get("kind", ""), f.get("statement", ""))


def _value_match(a: str, b: str) -> bool:
    va, vb = normalize_value(a), normalize_value(b)
    if not va or not vb:
        return False
    if va == vb:
        return True
    short, long_ = sorted((va, vb), key=len)
    return len(short) >= 3 and short in long_


def _pair(ref: list[Fact], pred: list[Fact], same) -> list[tuple[int, int]]:
    """Pareamento guloso um para um, na ordem da referência."""
    used: set[int] = set()
    pairs = []
    for i, r in enumerate(ref):
        for j, p in enumerate(pred):
            if j not in used and same(r, p):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs


@dataclass
class CaseScore:
    id: str
    category: str
    json_ok: bool
    schema_ok: bool
    n_ref: int
    n_pred: int
    strict_tp: int
    value_tp: int
    kind_pairs: int
    kind_ok: int
    reuse_total: int  # fatos da referência que reutilizam chave existente
    reuse_ok: int
    pred_existing: int  # fatos previstos com chave existente
    false_reuse: int
    ref_empty: bool
    pred_empty: bool
    latency_s: float | None = None
    completion_tokens: int | None = None


def score_case(case_id: str, category: str, existing_keys: list[str], ref: list, parsed: Parsed,
               latency_s: float | None = None, completion_tokens: int | None = None) -> CaseScore:
    ref_f = [_as_fact(f) for f in ref]
    pred_f = parsed.facts if parsed.schema_ok else []
    strict = _pair(ref_f, pred_f, lambda r, p: normalize_key(r.subject_key) == normalize_key(p.subject_key)
                   and normalize_value(r.value) == normalize_value(p.value))
    by_value = _pair(ref_f, pred_f, lambda r, p: _value_match(r.value, p.value))
    kind_ok = sum(ref_f[i].kind == pred_f[j].kind for i, j in by_value)

    existing = {k.strip() for k in existing_keys}
    ref_keys = {f.subject_key for f in ref_f}
    pred_keys = {f.subject_key for f in pred_f}
    reuse_refs = [f for f in ref_f if f.subject_key in existing]
    reuse_ok = sum(f.subject_key in pred_keys for f in reuse_refs)
    pred_existing = [f for f in pred_f if f.subject_key in existing]
    false_reuse = sum(f.subject_key not in ref_keys for f in pred_existing)

    return CaseScore(
        case_id, category, parsed.json_ok, parsed.schema_ok, len(ref_f), len(pred_f),
        len(strict), len(by_value), len(by_value), kind_ok, len(reuse_refs), reuse_ok,
        len(pred_existing), false_reuse, not ref_f, parsed.schema_ok and not pred_f,
        latency_s, completion_tokens,
    )


def _prf(tp: int, n_pred: int, n_ref: int) -> tuple[float | None, float | None, float | None]:
    p = tp / n_pred if n_pred else None
    r = tp / n_ref if n_ref else None
    if p is None or r is None:
        return p, r, None
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def aggregate(scores: list[CaseScore]) -> dict:
    """Médias micro sobre os casos. None quando a métrica não se aplica (denominador zero)."""
    n = len(scores)
    out: dict = {"n": n}
    if not n:
        return out
    out["json_valid"] = mean(s.json_ok for s in scores)
    out["schema_valid"] = mean(s.schema_ok for s in scores)
    n_pred = sum(s.n_pred for s in scores)
    n_ref = sum(s.n_ref for s in scores)
    out["strict_precision"], out["strict_recall"], out["strict_f1"] = _prf(
        sum(s.strict_tp for s in scores), n_pred, n_ref)
    out["value_precision"], out["value_recall"], out["value_f1"] = _prf(
        sum(s.value_tp for s in scores), n_pred, n_ref)
    out["kind_accuracy"] = _ratio(sum(s.kind_ok for s in scores), sum(s.kind_pairs for s in scores))
    out["reuse_accuracy"] = _ratio(sum(s.reuse_ok for s in scores), sum(s.reuse_total for s in scores))
    out["false_reuse_rate"] = _ratio(sum(s.false_reuse for s in scores), sum(s.pred_existing for s in scores))
    empties = [s for s in scores if s.ref_empty]
    out["empty_accuracy"] = _ratio(sum(s.pred_empty for s in empties), len(empties))
    nonempty = [s for s in scores if not s.ref_empty]
    out["false_empty_rate"] = _ratio(sum(s.pred_empty for s in nonempty), len(nonempty))
    lat = [s.latency_s for s in scores if s.latency_s is not None]
    toks = [s.completion_tokens for s in scores if s.completion_tokens is not None and s.latency_s is not None]
    if lat:
        out["latency_mean_s"] = mean(lat)
        out["latency_p50_s"] = median(lat)
        out["latency_p95_s"] = _pct(lat, 0.95)
    if toks:
        secs = sum(s.latency_s for s in scores if s.completion_tokens is not None and s.latency_s is not None)
        out["tokens_per_s"] = sum(toks) / secs if secs else None
        out["completion_tokens_mean"] = mean(toks)
    return out


def by_category(scores: list[CaseScore]) -> dict[str, dict]:
    groups: dict[str, list[CaseScore]] = defaultdict(list)
    for s in scores:
        groups[s.category].append(s)
    return {c: aggregate(g) for c, g in sorted(groups.items())}


def scores_to_rows(scores: list[CaseScore]) -> list[dict]:
    return [asdict(s) for s in scores]


@dataclass
class Report:
    """Métricas de um modelo num conjunto de teste."""

    model: str
    dataset: str
    mode: str  # guided | free
    overall: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)


HEADLINE = ["schema_valid", "strict_f1", "value_f1", "reuse_accuracy", "kind_accuracy",
            "empty_accuracy", "false_empty_rate", "latency_p50_s", "tokens_per_s"]


def markdown_table(reports: list[Report], cols: list[str] = HEADLINE) -> str:
    def fmt(c: str, v) -> str:
        if v is None:
            return "-"
        if c.endswith("_s"):
            return f"{v:.2f}"
        if c == "tokens_per_s":
            return f"{v:.0f}"
        return f"{v:.1%}"

    head = "| modelo | conjunto | modo | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 3)
    lines = [head, sep]
    for r in reports:
        lines.append(f"| {r.model} | {r.dataset} | {r.mode} | "
                     + " | ".join(fmt(c, r.overall.get(c)) for c in cols) + " |")
    return "\n".join(lines)
