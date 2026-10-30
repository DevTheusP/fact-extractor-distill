"""Junta os relatórios de avaliação e do benchmark numa tabela markdown e em gráficos.

Uso:
    python -m distill.report --eval-dir results/eval --bench-dir results/bench --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import Report, markdown_table

# Ordem fixa das cores por modelo (paleta categórica validada: azul, laranja, verde-água).
# A cor segue o modelo, não a posição: um modelo ausente não repinta os outros.
MODEL_ORDER = ["aluno-base", "aluno-destilado", "professor"]
COLORS = {"aluno-base": "#2a78d6", "aluno-destilado": "#eb6834", "professor": "#1baf7a"}
EXTRA_COLORS = ["#eda100", "#e87ba4"]  # aluno 1.5B, se houver
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PUBLISHED = {"v1_naive": 0.62, "v2_keyed": 0.98}  # rodada publicada do agent-memory-bench (Ollama)


def load_reports(eval_dir: Path) -> list[Report]:
    out = []
    for p in sorted(eval_dir.glob("report_*.json")):
        out += [Report(**d) for d in json.loads(p.read_text())]
    return out


def load_bench(bench_dir: Path) -> dict[str, dict]:
    return {json.loads(p.read_text())["name"]: json.loads(p.read_text())
            for p in sorted(bench_dir.glob("headline_*.json"))}


def _order(names) -> list[str]:
    known = [m for m in MODEL_ORDER if m in names]
    return known + sorted(n for n in names if n not in MODEL_ORDER)


def _color(name: str, names: list[str]) -> str:
    if name in COLORS:
        return COLORS[name]
    extras = [n for n in names if n not in COLORS]
    return EXTRA_COLORS[extras.index(name) % len(EXTRA_COLORS)]


def pick(reports: list[Report], dataset: str, mode: str) -> list[Report]:
    sel = [r for r in reports if r.dataset == dataset and r.mode == mode]
    names = _order({r.model for r in sel})
    return sorted(sel, key=lambda r: names.index(r.model))


def bench_table(bench: dict[str, dict]) -> str:
    cols = ["retrieval", "ret_update", "ret_scope", "ret_rare_fact", "ret_injection"]
    lines = ["| extrator | memória | " + " | ".join(cols) + " | falhas de JSON |", "|" + "---|" * (len(cols) + 3)]
    for name in _order(bench):
        b = bench[name]
        for version in ("v1_naive", "v2_keyed", "v3_decay_rank", "v4_decay_tier"):
            h = b["headline"].get(version)
            if not h:
                continue
            cells = ["-" if h.get(c) is None else f"{h[c]:.1%}" for c in cols]
            lines.append(f"| {name} | {version} | " + " | ".join(cells) + f" | {b['extractor_failures']} |")
    lines.append(f"| publicado (Ollama, qwen2.5:7b) | v2_keyed | {PUBLISHED['v2_keyed']:.0%} | | | | | |")
    lines.append(f"| publicado (Ollama, qwen2.5:7b) | v1_naive | {PUBLISHED['v1_naive']:.0%} | | | | | |")
    return "\n".join(lines)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metrics(reports: list[Report], dataset: str, mode: str, out: Path,
                 metrics=("schema_valid", "strict_f1", "value_f1", "reuse_accuracy", "empty_accuracy")) -> Path:
    import matplotlib.pyplot as plt

    sel = pick(reports, dataset, mode)
    names = [r.model for r in sel]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    _style(ax)
    width = 0.8 / max(1, len(sel))
    for i, r in enumerate(sel):
        xs = [j + i * width for j in range(len(metrics))]
        ys = [r.overall.get(m) or 0 for m in metrics]
        bars = ax.bar(xs, ys, width * 0.92, color=_color(r.model, names), label=r.model)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.01, f"{y:.0%}", ha="center", va="bottom",
                    fontsize=8, color=INK_2)
    ax.set_xticks([j + width * (len(sel) - 1) / 2 for j in range(len(metrics))])
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"{dataset} · modo {mode}", color=INK, loc="left")
    ax.legend(frameon=False, ncols=len(sel), loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    path = out / f"metricas_{dataset.replace('|', '_').replace('=', '-')}_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latency(reports: list[Report], dataset: str, mode: str, out: Path) -> Path:
    import matplotlib.pyplot as plt

    sel = pick(reports, dataset, mode)
    names = [r.model for r in sel]
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=SURFACE)
    _style(ax)
    ys = [r.overall.get("latency_p50_s") or 0 for r in sel]
    bars = ax.bar(names, ys, 0.6, color=[_color(n, names) for n in names])
    for b, r, y in zip(bars, sel, ys):
        tps = r.overall.get("tokens_per_s")
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}s\n{tps:.0f} tok/s" if tps else f"{y:.2f}s",
                ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylabel("latência p50 por chamada (s)", color=INK_2)
    ax.set_title("Latência, um pedido por vez (T4)", color=INK, loc="left")
    fig.tight_layout()
    path = out / "latencia.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bench(bench: dict[str, dict], out: Path) -> Path:
    import matplotlib.pyplot as plt

    names = _order(bench)
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    _style(ax)
    ys = [bench[n]["headline"]["v2_keyed"]["retrieval"] for n in names]
    bars = ax.bar(names, ys, 0.6, color=[_color(n, names) for n in names])
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.01, f"{y:.0%}", ha="center", va="bottom",
                fontsize=9, color=INK_2)
    naive = [bench[n]["headline"]["v1_naive"]["retrieval"] for n in names]
    if naive:
        ax.axhline(naive[0], color=INK_2, linestyle="--", linewidth=1)
        ax.text(len(names) - 0.5, naive[0] + 0.01, f"busca ingênua (v1) {naive[0]:.0%}", ha="right",
                va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Recuperação média da v2_keyed por extrator", color=INK, loc="left")
    fig.tight_layout()
    path = out / "benchmark_recuperacao.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_results(eval_dir: Path, bench_dir: Path, out: Path, plots: bool = True) -> str:
    out.mkdir(parents=True, exist_ok=True)
    reports = load_reports(eval_dir)
    bench = load_bench(bench_dir) if bench_dir.exists() else {}
    parts = ["# Resultados\n"]
    for dataset in sorted({r.dataset for r in reports}):
        for mode in ("guided", "free"):
            sel = pick(reports, dataset, mode)
            if sel:
                parts += [f"## {dataset} · {mode}\n", markdown_table(sel), ""]
                if plots:
                    plot_metrics(reports, dataset, mode, out)
    if plots and pick(reports, "test_bench|ref=professor", "guided"):
        plot_latency(reports, "test_bench|ref=professor", "guided", out)
    if bench:
        parts += ["## agent-memory-bench, camada 1 (recuperação)\n", bench_table(bench), ""]
        if plots and all("v2_keyed" in b["headline"] for b in bench.values()):
            plot_bench(bench, out)
    text = "\n".join(parts)
    (out / "RESULTS.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    ap.add_argument("--bench-dir", type=Path, default=Path("results/bench"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    print(write_results(args.eval_dir, args.bench_dir, args.out, plots=not args.no_plots))


if __name__ == "__main__":
    main()
