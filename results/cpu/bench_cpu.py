"""Latência do GGUF q4_K_M só em CPU, nas chamadas reais do agent-memory-bench.

Usa as 160 entradas (mensagem + chaves existentes) que o aluno destilado recebeu
no benchmark do Kaggle (extract_cache_aluno-destilado.json), o mesmo SYSTEM e o
mesmo SCHEMA. Fala com um llama-server já rodando (OpenAI-compatível) e grava
uma linha por chamada em cpu_runs_<label>.jsonl e o resumo em cpu_summary_<label>.json.

    python results/cpu/bench_cpu.py --url http://localhost:8080 --label t8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from distill.prompt import SCHEMA, build_messages

HERE = Path(__file__).parent
CACHE = HERE.parent / "kaggle-run2/results/bench/extract_cache_aluno-destilado.json"


def call(url: str, text: str, keys: list[str], cache_prompt: bool = True) -> dict:
    body = {
        "messages": build_messages(text, keys),
        "temperature": 0,
        "max_tokens": 256,
        "cache_prompt": cache_prompt,  # False: refaz o prefill do SYSTEM a cada chamada
        "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "schema": SCHEMA}},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return {"wall_s": time.perf_counter() - t0, "out": out}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-cache", action="store_true", help="sem reaproveitar o prefixo do prompt")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    call(args.url, cache[0][0][0], cache[0][0][1])  # aquecimento, fora da conta

    rows, same = [], 0
    with (HERE / f"cpu_runs_{args.label}.jsonl").open("w") as f:
        for (text, keys), gpu_facts in cache:
            r = call(args.url, text, keys, cache_prompt=not args.no_cache)
            t = r["out"]["timings"]
            facts = json.loads(r["out"]["choices"][0]["message"]["content"])["facts"]
            same += facts == gpu_facts
            row = {"wall_s": r["wall_s"], "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
                   "gen_n": t["predicted_n"], "gen_ms": t["predicted_ms"], "n_facts": len(facts)}
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    wall = [r["wall_s"] for r in rows]
    summary = {
        "label": args.label,
        "prompt_cache": not args.no_cache,
        "calls": len(rows),
        "latency_p50_s": statistics.median(wall),
        "latency_p90_s": pct(wall, 0.9),
        "latency_max_s": max(wall),
        "latency_p50_s_empty": statistics.median([r["wall_s"] for r in rows if r["n_facts"] == 0]),
        "latency_p50_s_with_facts": statistics.median([r["wall_s"] for r in rows if r["n_facts"] > 0]),
        "prompt_tokens_p50": statistics.median(r["prompt_n"] for r in rows),
        "prefill_tok_s": sum(r["prompt_n"] for r in rows) / sum(r["prompt_ms"] for r in rows) * 1e3,
        "decode_tok_s": sum(r["gen_n"] for r in rows) / sum(r["gen_ms"] for r in rows) * 1e3,
        "same_output_as_t4": f"{same}/{len(rows)}",
    }
    (HERE / f"cpu_summary_{args.label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
