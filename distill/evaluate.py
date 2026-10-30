"""Avaliação de um modelo servido (aluno base, aluno destilado ou professor).

Para cada conjunto de teste rotulado (data/labeled/test_*.jsonl):
  1. vazão: todos os exemplos com pedidos concorrentes;
  2. latência: uma amostra com um pedido por vez (batch 1, como o benchmark usa);
  3. métricas contra três referências:
       professor        rótulo do Qwen2.5-7B-AWQ no Kaggle (fidelidade da destilação)
       gabarito         intenção do template / gabarito do cenário (independe do professor)
       professor_ollama só no test_bench: rótulo da rodada publicada (qwen2.5:7b q4 no Ollama)

Modos: guided (gramática do schema, como no benchmark) e free (sem gramática:
mede se o modelo aprendeu o formato).

Uso:
    python -m distill.evaluate --base-url http://127.0.0.1:8000 --model student --name aluno-destilado \\
        --labeled-dir data/labeled --out results/eval
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from .bench_data import ollama_labels
from .engines import Engine, OpenAIChatEngine, run_many
from .generate import read_jsonl, write_jsonl
from .label import drop_optional, gold_reference
from .metrics import Report, aggregate, by_category, score_case
from .parsing import parse_output
from .prompt import SCHEMA, build_messages


def predict(engine: Engine, rows: list[dict], guided: bool, workers: int, latency_sample: int,
            seed: int = 0) -> tuple[list[dict], dict]:
    msgs = [build_messages(r["text"], r["existing_keys"]) for r in rows]
    schema = SCHEMA if guided else None
    t0 = time.perf_counter()
    comps = run_many(engine, msgs, schema, workers=workers)
    wall = time.perf_counter() - t0
    tokens = sum(c.completion_tokens or 0 for c in comps)
    throughput = {"wall_s": wall, "requests_per_s": len(rows) / wall if wall else None,
                  "tokens_per_s_concurrent": tokens / wall if wall else None}

    # latência de um pedido por vez numa amostra fixa
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(latency_sample, len(rows))))
    lat = {}
    if idx:
        seq = run_many(engine, [msgs[i] for i in idx], schema, workers=1, progress=False)
        lat = {i: c for i, c in zip(idx, seq)}

    preds = []
    for i, (r, c) in enumerate(zip(rows, comps)):
        lc = lat.get(i)
        preds.append({"id": r["id"], "raw": c.text, "error": c.error,
                      "latency_s": lc.latency_s if lc else None,
                      "completion_tokens": lc.completion_tokens if lc else c.completion_tokens})
    return preds, throughput


def score(rows: list[dict], preds: list[dict], reference: str) -> tuple[dict, dict]:
    ollama = ollama_labels() if reference == "professor_ollama" else None
    scores = []
    for r, p in zip(rows, preds):
        parsed = parse_output(p["raw"])
        if reference == "professor":
            ref = r["teacher"]["facts"]
        elif reference == "gabarito":
            ref, optional = gold_reference(r)
            parsed.facts = drop_optional(parsed.facts, optional)
        else:
            ref = ollama[(r["text"], tuple(r["existing_keys"]))]
        scores.append(score_case(r["id"], r["category"], r["existing_keys"], ref, parsed,
                                 p["latency_s"], p["completion_tokens"]))
    return aggregate(scores), by_category(scores)


def evaluate(engine: Engine, name: str, labeled_dir: Path, splits: list[str], modes: list[str],
             out_dir: Path, workers: int = 32, latency_sample: int = 50, limit: int | None = None) -> list[Report]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Report] = []
    for split in splits:
        rows = read_jsonl(labeled_dir / f"{split}.jsonl")
        rows = rows[:limit] if limit else rows
        for mode in modes:
            print(f"{name} | {split} | {mode}: {len(rows)} exemplos", flush=True)
            preds, thr = predict(engine, rows, mode == "guided", workers, latency_sample)
            write_jsonl(out_dir / f"pred_{name}_{split}_{mode}.jsonl", preds)
            refs = ["professor", "gabarito"] + (["professor_ollama"] if split == "test_bench" else [])
            for ref in refs:
                overall, cats = score(rows, preds, ref)
                overall.update(thr)
                reports.append(Report(name, f"{split}|ref={ref}", mode, overall, cats))
    (out_dir / f"report_{name}.json").write_text(
        json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=2))
    return reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="served-model-name no vLLM")
    ap.add_argument("--name", required=True, help="rótulo do modelo nas tabelas")
    ap.add_argument("--labeled-dir", type=Path, default=Path("data/labeled"))
    ap.add_argument("--splits", nargs="+", default=["test_synth", "test_bench"])
    ap.add_argument("--modes", nargs="+", default=["guided", "free"], choices=["guided", "free"])
    ap.add_argument("--out", type=Path, default=Path("results/eval"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--latency-sample", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    engine = OpenAIChatEngine(args.base_url, args.model, name=args.name)
    evaluate(engine, args.name, args.labeled_dir, args.splits, args.modes, args.out,
             args.workers, args.latency_sample, args.limit)


if __name__ == "__main__":
    main()
