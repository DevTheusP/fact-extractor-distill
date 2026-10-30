"""Camada 1 (recuperação) do agent-memory-bench com outro extrator no lugar do professor.

Reaproveita o runner do benchmark sem modificá-lo: só troca o OllamaLLM pelo
EngineLLM (o modelo servido no vLLM) e o OllamaEmbedder pelo bge-m3 via
sentence-transformers. As versões v2 a v4 dependem do extrator; v1 (busca
ingênua) não usa LLM e serve de controle: ela tem que dar o mesmo número com
qualquer extrator.

Uso (com o servidor do modelo no ar e o benchmark clonado):
    python -m distill.bench --bench-repo ../agent-memory-bench --base-url http://127.0.0.1:8000 \\
        --model student --name aluno-destilado --out results/bench
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

PUBLISHED_GRID = {"seeds": [1, 2, 3, 4, 5], "noise_levels": [3, 10, 25, 50, 100],
                  "gap_days": [0, 90, 180, 365, 730]}
SCENARIOS = ["update", "scope", "rare_fact", "injection"]


def import_bench(repo: Path):
    """Importa os módulos do benchmark a partir do clone (eles não são um pacote instalável)."""
    repo = Path(repo).resolve()
    if not (repo / "eval" / "run.py").exists():
        raise FileNotFoundError(f"{repo} não parece o agent-memory-bench (falta eval/run.py)")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import eval.run as run  # noqa: PLC0415
    from memory.extract import LLMExtractor  # noqa: PLC0415
    from scenarios.build import build_timeline  # noqa: PLC0415

    return run, LLMExtractor, build_timeline


def run_layer1(llm, embedder, repo: Path, out_dir: Path, name: str, seeds: list[int],
               noise_levels: list[int], gap_days: list[float]) -> list[dict]:
    run, LLMExtractor, build_timeline = import_bench(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = run.CachedEmbedder(embedder)
    extractor = run.CachedExtractor(LLMExtractor(llm), path=out_dir / f"extract_cache_{name}.json")
    rows = []
    with (out_dir / f"trace_{name}.jsonl").open("w") as trace:
        for gap in gap_days:
            for noise in noise_levels:
                for seed in seeds:
                    tl = build_timeline(seed, noise, gap)
                    for policy in run.make_policies(emb, extractor, top_k=5, include_baseline=True):
                        rows += run.run_version(policy, tl, trace)
                print(f"  pausa {gap:g}d, ruído {noise}/sem: ok", flush=True)
    dicts = [asdict(r) for r in rows]
    run._write_csv(out_dir / f"runs_{name}.csv", dicts)
    summary = run.summarize(rows)
    run._write_csv(out_dir / f"summary_{name}.csv", summary)
    return summary


def headline(summary: list[dict]) -> dict[str, dict]:
    """Média sobre todas as configurações (pausa x ruído), por versão de memória."""
    out: dict[str, dict] = {}
    for version in sorted({s["version"] for s in summary}):
        lines = [s for s in summary if s["version"] == version]
        d = {"retrieval": mean(s["retrieval"] for s in lines)}
        for sc in SCENARIOS:
            vals = [s[f"ret_{sc}"] for s in lines if s[f"ret_{sc}"] is not None]
            d[f"ret_{sc}"] = mean(vals) if vals else None
        out[version] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-repo", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--free", action="store_true", help="sem gramática (o padrão é guiado, como no Ollama)")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--embed-device", default=None, help="ex.: cuda:1 para não disputar a GPU do vLLM")
    ap.add_argument("--seeds", type=int, nargs="+", default=PUBLISHED_GRID["seeds"])
    ap.add_argument("--noise-levels", type=int, nargs="+", default=PUBLISHED_GRID["noise_levels"])
    ap.add_argument("--gap-days", type=float, nargs="+", default=PUBLISHED_GRID["gap_days"])
    ap.add_argument("--out", type=Path, default=Path("results/bench"))
    args = ap.parse_args()

    from .adapter import EngineLLM, SentenceTransformerEmbedder
    from .engines import OpenAIChatEngine

    llm = EngineLLM(OpenAIChatEngine(args.base_url, args.model, name=args.name), guided=not args.free)
    embedder = SentenceTransformerEmbedder(args.embed_model, device=args.embed_device)
    summary = run_layer1(llm, embedder, args.bench_repo, args.out, args.name,
                         args.seeds, args.noise_levels, args.gap_days)
    result = {"name": args.name, "guided": not args.free, "extractor_calls": llm.calls,
              "extractor_failures": len(llm.failures), "headline": headline(summary)}
    (args.out / f"headline_{args.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (args.out / f"failures_{args.name}.json").write_text(json.dumps(llm.failures, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
