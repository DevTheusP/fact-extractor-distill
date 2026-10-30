"""Rotulagem pelo professor e montagem do dataset de SFT.

O professor (Qwen2.5-7B-Instruct-AWQ no vLLM) recebe exatamente o SYSTEM, o
SCHEMA e a mensagem de usuário do benchmark, com saída guiada pelo schema e
temperatura 0. A resposta dele é o alvo do aluno: destilação por rótulo
(sequence-level), não por logits.

Uso (com o servidor do professor no ar):
    python -m distill.label --base-url http://127.0.0.1:8000 --model teacher \\
        --data-dir data/generated --out-dir data/labeled --bench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bench_data import bench_test_cases
from .engines import Engine, OpenAIChatEngine, run_many
from .formatting import sft_record
from .generate import Example, read_jsonl, write_jsonl
from .metrics import aggregate, by_category, score_case
from .parsing import Fact, parse_output
from .prompt import SCHEMA, build_messages


def label_examples(engine: Engine, examples: list[Example], guided: bool = True, workers: int = 32,
                   max_tokens: int = 512) -> list[dict]:
    batch = [build_messages(e.text, e.existing_keys) for e in examples]
    comps = run_many(engine, batch, SCHEMA if guided else None, max_tokens, workers=workers)
    rows = []
    for e, c in zip(examples, comps):
        p = parse_output(c.text) if not c.error else None
        rows.append({
            **e.to_dict(),
            "teacher": {
                "model": engine.name,
                "guided": guided,
                "raw": c.text,
                "json_ok": bool(p and p.json_ok),
                "schema_ok": bool(p and p.schema_ok),
                "facts": [f.to_dict() for f in p.facts] if p and p.schema_ok else [],
                "error": c.error or (p.error if p else None),
                "latency_s": c.latency_s,
                "completion_tokens": c.completion_tokens,
            },
        })
    return rows


def build_sft(rows: list[dict]) -> tuple[list[dict], dict]:
    """Só entram rótulos válidos. Devolve os registros e quantos ficaram de fora."""
    records, dropped = [], 0
    for r in rows:
        t = r["teacher"]
        if not t["schema_ok"]:
            dropped += 1
            continue
        records.append(sft_record(r["text"], r["existing_keys"], t["facts"], r["id"]))
    return records, {"kept": len(records), "dropped_invalid": dropped}


def gold_reference(row: dict) -> tuple[list[dict], list[str]]:
    """Gabarito do template (ou do cenário do benchmark): fatos obrigatórios e valores opcionais.

    Valores opcionais (reforço de fato já conhecido) não contam a favor nem contra.
    """
    ref, optional = [], []
    for i in row.get("intents", []):
        if i.get("optional"):
            if i.get("value"):
                optional.append(i["value"])
            continue
        ref.append({"subject_key": i["key"], "value": i["value"], "kind": i["kind"], "statement": ""})
    return ref, optional


def drop_optional(facts: list[Fact], optional: list[str]) -> list[Fact]:
    from .metrics import _value_match

    return [f for f in facts if not any(_value_match(f.value, v) for v in optional)]


def teacher_vs_gold(rows: list[dict]) -> dict:
    """Qualidade dos rótulos: o professor concorda com a intenção do template?"""
    scores = []
    for r in rows:
        ref, optional = gold_reference(r)
        p = parse_output(r["teacher"]["raw"])
        p.facts = drop_optional(p.facts, optional)
        scores.append(score_case(r["id"], r["category"], r["existing_keys"], ref, p))
    return {"overall": aggregate(scores), "categories": by_category(scores)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="teacher", help="served-model-name no vLLM")
    ap.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test_synth"])
    ap.add_argument("--bench", action="store_true", help="também rotula os pares do agent-memory-bench")
    ap.add_argument("--out-dir", type=Path, default=Path("data/labeled"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="só os N primeiros de cada split (smoke test)")
    args = ap.parse_args()

    engine = OpenAIChatEngine(args.base_url, args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sets = {s: [Example.from_dict(d) for d in read_jsonl(args.data_dir / f"{s}.jsonl")] for s in args.splits}
    if args.bench:
        sets["test_bench"] = bench_test_cases()
    report = {}
    for name, examples in sets.items():
        examples = examples[: args.limit] if args.limit else examples
        print(f"rotulando {name}: {len(examples)} exemplos", flush=True)
        rows = label_examples(engine, examples, workers=args.workers)
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        report[name] = {
            "n": len(rows),
            "schema_valid": sum(r["teacher"]["schema_ok"] for r in rows) / max(1, len(rows)),
            "empty_rate": sum(not r["teacher"]["facts"] for r in rows) / max(1, len(rows)),
            "teacher_vs_gold": teacher_vs_gold(rows),
        }
        if name in ("train", "val"):
            records, stats = build_sft(rows)
            write_jsonl(args.out_dir / f"sft_{name}.jsonl", records)
            report[name]["sft"] = stats
        print(json.dumps({k: v for k, v in report[name].items() if k != "teacher_vs_gold"}, ensure_ascii=False))
    (args.out_dir / "label_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
