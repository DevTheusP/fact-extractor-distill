"""Gera notebooks/fact_extractor_distill.ipynb a partir das células abaixo.

O notebook é derivado deste arquivo: edite aqui e rode
    python notebooks/build_notebook.py
O teste tests/test_notebook.py confere que o .ipynb está em dia com este script.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("fact_extractor_distill.ipynb")

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Destilando um extrator de memória: Qwen2.5-7B → Qwen2.5-0.5B

**Pergunta:** um modelo 14x menor consegue fazer o papel de extrator de fatos da memória de um agente?

No [agent-memory-bench](https://github.com/DevTheusP/agent-memory-bench), a política de memória `v2_keyed`
usa um LLM (Qwen2.5-7B) para extrair fatos duráveis de cada mensagem, com uma chave de sujeito
(`subject_key`) que resolve atualizações na escrita. Com ela, a recuperação média foi de 98%, contra 62%
da busca ingênua. Este notebook destila esse extrator para o Qwen2.5-0.5B-Instruct com LoRA e mede se o
aluno preserva o resultado.

Etapas:
1. gera ~3 mil mensagens sintéticas em PT-BR (templates com seed fixa, sem nada do benchmark);
2. o professor (Qwen2.5-7B-Instruct-AWQ no vLLM) rotula com o mesmo prompt e schema do benchmark;
3. treina o aluno com LoRA (TRL + PEFT);
4. compara aluno base, aluno destilado e professor: JSON válido, F1 de fatos, reutilização de chave, latência;
5. roda a camada 1 do benchmark com cada extrator;
6. exporta adaptador, modelo merged e GGUF q4 (Ollama) e, se quiser, publica no Hugging Face.

Código e testes: [fact-extractor-distill](https://github.com/DevTheusP/fact-extractor-distill).
Passo a passo para rodar no Kaggle: `KAGGLE.md` no repositório.
""")

md("## 1. Configuração")

code('''
# ---- edite aqui -------------------------------------------------------------
SMOKE_TEST = False          # True: pipeline inteiro com ~20 exemplos, para conferir que tudo roda

STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ALSO_STUDENT_1_5B = False   # também destila para o Qwen2.5-1.5B-Instruct (+~1 h)
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

N_TRAIN, N_VAL, N_TEST = 4000, 200, 300  # ~20% dos rótulos caem nos verificadores
EPOCHS = 2
LEARNING_RATE = 2e-4
LORA_R, LORA_ALPHA = 16, 32
BATCH_SIZE, GRAD_ACCUM = 8, 2

RUN_BENCH = True            # camada 1 do agent-memory-bench com cada extrator
BENCH_GRID = "published"    # "published" (5 seeds x 5 ruídos x 5 pausas) ou "small"
EXPORT_GGUF = True          # GGUF q4_K_M + Modelfile para o Ollama
PUSH_TO_HUB = False         # precisa do secret HF_TOKEN (Add-ons > Secrets)
HF_REPO_MODEL = "DevTheusP/qwen2.5-0.5b-fact-extractor-ptbr"
HF_REPO_DATASET = "DevTheusP/fact-extractor-distill-ptbr"

REPO_URL = "https://github.com/DevTheusP/fact-extractor-distill.git"
BENCH_URL = "https://github.com/DevTheusP/agent-memory-bench.git"
VLLM_VERSION = "0.32.0"     # conferida no PyPI em 2026-10-30; ver KAGGLE.md se não subir na T4
VLLM_ENFORCE_EAGER = False  # True se a captura de CUDA graphs falhar ou estourar memória
# -----------------------------------------------------------------------------

MAX_STEPS = -1
if SMOKE_TEST:
    N_TRAIN, N_VAL, N_TEST = 20, 8, 10
    EPOCHS, MAX_STEPS = 1, 10
    BENCH_GRID = "small"
    ALSO_STUDENT_1_5B = False

GRIDS = {
    "published": dict(seeds=[1, 2, 3, 4, 5], noise=[3, 10, 25, 50, 100], gaps=[0, 90, 180, 365, 730]),
    "small": dict(seeds=[1], noise=[3], gaps=[0]),
}
print("SMOKE_TEST =", SMOKE_TEST, "| aluno:", STUDENT_MODEL, "| treino:", N_TRAIN, "| grade:", BENCH_GRID)
''')

md("## 2. Ambiente: GPU, código e dependências")

code('''
import os, subprocess, sys, time, json, shutil
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
TMP = Path("/kaggle/tmp") if Path("/kaggle").exists() else WORK / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TMP / "hf")          # cache de modelos fora do /kaggle/working (limite de 20 GB)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TIMES = {}
def etapa(nome):
    """Marca o tempo de cada etapa para a tabela final."""
    TIMES[nome] = time.time()
    print(f"\\n=== {nome} ===", flush=True)

def fim(nome):
    TIMES[nome] = time.time() - TIMES[nome]
    print(f"--- {nome}: {TIMES[nome] / 60:.1f} min", flush=True)

def run(*args, env=None, check=True):
    """Roda um comando mostrando a saída ao vivo."""
    cmd = [str(a) for a in args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env={**os.environ, **(env or {})})

subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"])
import torch
N_GPUS = torch.cuda.device_count()
assert N_GPUS >= 1, "Sem GPU: ative Accelerator > GPU T4 x2 nas configurações do notebook."
print("GPUs:", N_GPUS, [torch.cuda.get_device_name(i) for i in range(N_GPUS)], "| torch", torch.__version__)
''')

code('''
etapa("instalação")
REPO = WORK / "fact-extractor-distill"
BENCH = WORK / "agent-memory-bench"
if not REPO.exists():
    run("git", "clone", "--depth", "1", REPO_URL, REPO)
if not BENCH.exists():
    run("git", "clone", "--depth", "1", BENCH_URL, BENCH)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# ambiente principal: treino, avaliação e benchmark (mantém o torch do Kaggle)
run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt")
run(sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps")
# o Kaggle traz torchao 0.10, que o PEFT 0.21 recusa (exige > 0.16); o projeto não usa torchao
run(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao", check=False)

# vLLM num venv separado: ele fixa a própria versão do torch
run(sys.executable, "-m", "pip", "install", "-q", "uv")
VLLM_ENV = TMP / "vllm-env"
VLLM_BIN = VLLM_ENV / "bin" / "vllm"
if not VLLM_BIN.exists():
    run("uv", "venv", VLLM_ENV, "--python", "3.12", "--seed")
    run("uv", "pip", "install", "--python", VLLM_ENV / "bin" / "python", f"vllm=={VLLM_VERSION}")
run(VLLM_ENV / "bin" / "python", "-c", "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)")
fim("instalação")
''')

md("""
## 3. Dados sintéticos

Templates + combinatória com seed fixa (justificativa no README): 12 categorias, de configuração e
regras a ruído do dia a dia, mensagens que negam o que dizem, conteúdo externo e listas de chaves
existentes que devem (ou não) ser reutilizadas. O teste sintético usa áreas e regras que não aparecem
no treino. Nada dos cenários do benchmark entra aqui: `distill/leakage.py` confere.
""")

code('''
etapa("geração")
run(sys.executable, "-m", "distill.generate", "--out", "data/generated",
    "--n-train", N_TRAIN, "--n-val", N_VAL, "--n-test", N_TEST)
from distill.generate import read_jsonl
for row in read_jsonl("data/generated/train.jsonl")[:6]:
    print(f"[{row['category']}] {row['text']}  | chaves: {row['existing_keys']}")
fim("geração")
''')

md("""
## 4. Professor: rotulagem com o Qwen2.5-7B-Instruct-AWQ

O vLLM sobe como servidor compatível com a API da OpenAI e cada pedido usa saída guiada pelo `SCHEMA`
do benchmark (`response_format: json_schema`), o equivalente ao `format` do Ollama na rodada publicada.
Na mesma subida, o professor também é avaliado e roda o benchmark.
""")

code('''
from distill.server import VLLMServer

def bench_args(name, base_url, model):
    g = GRIDS[BENCH_GRID]
    embed_device = "cuda:1" if N_GPUS > 1 else "cpu"   # a GPU 0 é do vLLM
    return [sys.executable, "-m", "distill.bench", "--bench-repo", BENCH, "--base-url", base_url,
            "--model", model, "--name", name, "--out", "results/bench", "--embed-device", embed_device,
            "--seeds", *g["seeds"], "--noise-levels", *g["noise"], "--gap-days", *g["gaps"]]

def evaluate_served(name, base_url, model):
    run(sys.executable, "-m", "distill.evaluate", "--base-url", base_url, "--model", model, "--name", name,
        "--labeled-dir", "data/labeled", "--out", "results/eval",
        "--latency-sample", 10 if SMOKE_TEST else 50)
    if RUN_BENCH:
        run(*bench_args(name, base_url, model))

etapa("professor")
with VLLMServer(VLLM_BIN, TEACHER_MODEL, "teacher", gpu="0", log_dir=WORK / "logs",
                gpu_memory_utilization=0.88, max_model_len=4096, enforce_eager=VLLM_ENFORCE_EAGER) as url:
    run(sys.executable, "-m", "distill.label", "--base-url", url, "--model", "teacher",
        "--data-dir", "data/generated", "--out-dir", "data/labeled", "--bench")
    evaluate_served("professor", url, "teacher")
fim("professor")

report = json.loads(Path("data/labeled/label_report.json").read_text())
for split, r in report.items():
    gold = r["teacher_vs_gold"]["overall"]
    print(f"{split:11s} n={r['n']:5d} schema ok={r['schema_valid']:.1%} vazios={r['empty_rate']:.1%} "
          f"| professor x template: value F1={gold.get('value_f1') or 0:.1%} "
          f"lista vazia certa={gold.get('empty_accuracy') or 0:.1%}")
''')

md("""
## 5. Dataset de SFT e conferência do chat template

O alvo é a resposta do professor, no formato conversacional prompt-completion do TRL (a perda fica só
no JSON da resposta). Antes de treinar, conferimos que a renderização em Python puro usada nos testes
locais bate com o `apply_chat_template` do tokenizer.
""")

code('''
from transformers import AutoTokenizer
from distill.formatting import render_record

sft = read_jsonl("data/labeled/sft_train.jsonl")
tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
diff = [rec["id"] for rec in sft[:50]
        if tok.apply_chat_template(rec["prompt"] + rec["completion"], tokenize=False) != render_record(rec)]
# o treino usa o template do tokenizer; a diferença só invalidaria os testes locais de formatação
print("chat template confere com distill/formatting.py" if not diff else f"ATENÇÃO: template diferente em {diff[:3]}")
lengths = [len(tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=True)) for r in sft]
print(f"{len(sft)} exemplos de treino; tokens por exemplo: média {sum(lengths)/len(lengths):.0f}, máx {max(lengths)}")
print(render_record(sft[0])[-400:])
''')

md("## 6. Aluno base (sem fine-tune)")

code('''
def serve_student(model_path, served):
    return VLLMServer(VLLM_BIN, str(model_path), served, gpu="0", log_dir=WORK / "logs",
                      gpu_memory_utilization=0.5, max_model_len=4096, enforce_eager=VLLM_ENFORCE_EAGER)

def short(model_id):
    return model_id.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "").lower()  # "0.5b"

def tag(prefix, model_id):
    return prefix if model_id == STUDENT_MODEL else f"{prefix}-{short(model_id)}"

STUDENTS = [STUDENT_MODEL] + (["Qwen/Qwen2.5-1.5B-Instruct"] if ALSO_STUDENT_1_5B else [])

for student in STUDENTS:
    etapa(f"aluno base {short(student)}")
    with serve_student(student, "student") as url:
        evaluate_served(tag("aluno-base", student), url, "student")
    fim(f"aluno base {short(student)}")
''')

md("""
## 7. Treino LoRA (TRL + PEFT)

LoRA r=16 em todas as projeções lineares, 2 épocas, perda só na resposta. A T4 não tem bfloat16:
o modelo carrega em float32 e o treino usa autocast fp16. O treino roda em subprocesso para liberar
a GPU ao terminar.
""")

code('''
MODELS = TMP / "models"
for student in STUDENTS:
    name = f"qwen2.5-{short(student)}-fact-extractor"
    out = MODELS / name
    etapa(f"treino {short(student)}")
    bs = BATCH_SIZE if "0.5" in student else max(2, BATCH_SIZE // 2)
    ga = GRAD_ACCUM if "0.5" in student else GRAD_ACCUM * 2
    run(sys.executable, "-m", "distill.train", "--student", student, "--data-dir", "data/labeled", "--out", out,
        "--epochs", EPOCHS, "--lr", LEARNING_RATE, "--lora-r", LORA_R, "--lora-alpha", LORA_ALPHA,
        "--batch-size", bs, "--grad-accum", ga, "--max-steps", MAX_STEPS, env={"CUDA_VISIBLE_DEVICES": "0"})
    fim(f"treino {short(student)}")
    etapa(f"merge {short(student)}")
    run(sys.executable, "-m", "distill.export", "merge", "--base", student, "--adapter", out / "adapter",
        "--out", out / "merged", env={"CUDA_VISIBLE_DEVICES": ""})
    fim(f"merge {short(student)}")
''')

code('''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.5))
for i, student in enumerate(STUDENTS):
    log = json.loads((MODELS / f"qwen2.5-{short(student)}-fact-extractor" / "train_log.json").read_text())
    hist = log["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    color = ["#2a78d6", "#eb6834"][i % 2]
    if tr:
        ax.plot(*zip(*tr), color=color, linewidth=2, label=f"{short(student)} treino")
    if ev:
        ax.plot(*zip(*ev), color=color, linewidth=2, linestyle="--", marker="o", markersize=4,
                label=f"{short(student)} validação")
    print(short(student), "| perda final de validação:", log["final_eval"].get("eval_loss"),
          "| tempo:", round((log["train_runtime_s"] or 0) / 60, 1), "min")
ax.set_xlabel("passo"); ax.set_ylabel("perda (só na resposta)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
''')

md("## 8. Aluno destilado")

code('''
for student in STUDENTS:
    merged = MODELS / f"qwen2.5-{short(student)}-fact-extractor" / "merged"
    etapa(f"aluno destilado {short(student)}")
    with serve_student(merged, "student") as url:
        evaluate_served(tag("aluno-destilado", student), url, "student")
    fim(f"aluno destilado {short(student)}")
''')

md("""
## 9. Resultados

Três referências: **professor** (o rótulo que o aluno imitou: fidelidade da destilação), **gabarito**
(a intenção do template ou do cenário, que vale também para o professor) e, no teste do benchmark,
**professor_ollama** (os rótulos da rodada publicada, qwen2.5:7b q4 no Ollama). Modo `guided` é com
gramática do schema, como no benchmark; `free` é sem gramática.
""")

code('''
from IPython.display import Markdown, Image, display
from distill.report import write_results

text = write_results(Path("results/eval"), Path("results/bench"), Path("results"))
display(Markdown(text))
for png in sorted(Path("results").glob("*.png")):
    if "ref-professor_" in png.name or png.name in ("latencia.png", "benchmark_recuperacao.png"):
        display(Image(str(png)))
''')

code('''
# alguns casos lado a lado no teste do benchmark (a mensagem da migração exige reutilizar a chave)
from distill.parsing import parse_output
rows = read_jsonl("data/labeled/test_bench.jsonl")
preds = {n: {p["id"]: p for p in read_jsonl(f"results/eval/pred_{n}_test_bench_guided.jsonl")}
         for n in ("aluno-base", "aluno-destilado")}
interesting = [r for r in rows if r["teacher"]["facts"]][:6]
for r in interesting:
    print("MENSAGEM:", r["text"]); print("  chaves existentes:", r["existing_keys"])
    print("  professor:", [(f["subject_key"], f["value"]) for f in r["teacher"]["facts"]])
    for n, p in preds.items():
        print(f"  {n}:", [(f.subject_key, f.value) for f in parse_output(p[r["id"]]["raw"]).facts])
    print()
''')

md("## 10. Exportação: GGUF q4 para Ollama")

code('''
if EXPORT_GGUF:
    etapa("gguf")
    LLAMA = TMP / "llama.cpp"
    GGUF_ENV = TMP / "gguf-env"
    if not LLAMA.exists():
        run("git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA)
    if shutil.which("cmake") is None:
        run(sys.executable, "-m", "pip", "install", "-q", "cmake")
    if not (LLAMA / "build" / "bin" / "llama-quantize").exists():
        run("cmake", "-S", LLAMA, "-B", LLAMA / "build", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF")
        run("cmake", "--build", LLAMA / "build", "--target", "llama-quantize", "-j", os.cpu_count())
    if not (GGUF_ENV / "bin" / "python").exists():
        run("uv", "venv", GGUF_ENV, "--python", "3.12", "--seed")
        run("uv", "pip", "install", "--python", GGUF_ENV / "bin" / "python",
            "-r", LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")
    for student in STUDENTS:
        base = MODELS / f"qwen2.5-{short(student)}-fact-extractor"
        run(sys.executable, "-m", "distill.export", "gguf", "--merged", base / "merged", "--out", base / "gguf",
            "--name", f"fact-extractor-{short(student)}", "--llama-cpp", LLAMA, "--python", GGUF_ENV / "bin" / "python")
        print((base / "gguf" / "Modelfile").read_text())
    fim("gguf")
''')

md("""
## 11. Publicação (opcional)

Só roda com `PUSH_TO_HUB = True`. O token vem dos Kaggle Secrets (`HF_TOKEN`) e vai só para a
variável de ambiente que o `huggingface_hub` lê; não aparece no notebook nem nos arquivos.
""")

code('''
from distill.export import assemble_hub_folder
from distill.report import load_bench, load_reports, pick
from distill.metrics import markdown_table

def fill_card(src, dst, table_md):
    text = Path(src).read_text()
    dst.write_text(text.replace("<!-- RESULTADOS -->", table_md))

reports = load_reports(Path("results/eval"))
table = markdown_table(pick(reports, "test_bench|ref=professor", "guided") +
                       pick(reports, "test_synth|ref=professor", "free"))
bench_md = (Path("results/RESULTS.md").read_text().split("## agent-memory-bench", 1)[-1]
            if Path("results/RESULTS.md").exists() else "")
results_md = table + "\\n\\n### agent-memory-bench" + bench_md

def label_quality_md():
    rep = json.loads(Path("data/labeled/label_report.json").read_text())
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    lines = ["| divisão | n | schema válido | listas vazias | F1 por valor vs. template | reutilização | lista vazia certa |",
             "|---|---|---|---|---|---|---|"]
    for split, r in rep.items():
        g = r["teacher_vs_gold"]["overall"]
        lines.append(f"| {split} | {r['n']} | {pct(r['schema_valid'])} | {pct(r['empty_rate'])} | "
                     f"{pct(g.get('value_f1'))} | {pct(g.get('reuse_accuracy'))} | {pct(g.get('empty_accuracy'))} |")
    return "\\n".join(lines)

if PUSH_TO_HUB:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    from distill.export import push
    base = MODELS / f"qwen2.5-{short(STUDENT_MODEL)}-fact-extractor"
    card = TMP / "MODEL_CARD.md"; fill_card("hf/MODEL_CARD.md", card, results_md)
    from distill.prompt import SCHEMA, SYSTEM
    prompt_file, schema_file = TMP / "system_prompt.txt", TMP / "schema.json"
    prompt_file.write_text(SYSTEM); schema_file.write_text(json.dumps(SCHEMA, indent=2))
    folder = assemble_hub_folder(TMP / "hub_model", base / "merged", base / "adapter",
                                 base / "gguf" if EXPORT_GGUF else None, card,
                                 extra={"train_log.json": base / "train_log.json",
                                        "system_prompt.txt": prompt_file, "schema.json": schema_file})
    print(push(folder, HF_REPO_MODEL))
    ds = TMP / "hub_dataset"; shutil.rmtree(ds, ignore_errors=True); (ds / "data").mkdir(parents=True)
    for f in ["train", "val", "test_synth", "test_bench", "sft_train", "sft_val"]:
        shutil.copy(f"data/labeled/{f}.jsonl", ds / "data" / f"{f}.jsonl")
    shutil.copy("data/labeled/label_report.json", ds / "label_report.json")
    # predições, relatórios e o que cada extrator escreveu no benchmark: a sessão do Kaggle apaga
    # /kaggle/working, o Hub não (os traces ficam de fora pelo tamanho)
    shutil.copytree("results/eval", ds / "results" / "eval", dirs_exist_ok=True)
    (ds / "results" / "bench").mkdir(parents=True, exist_ok=True)
    for p in Path("results/bench").glob("*.json"):
        shutil.copy(p, ds / "results" / "bench" / p.name)
    fill_card("hf/DATASET_CARD.md", ds / "README.md", label_quality_md())
    print(push(ds, HF_REPO_DATASET, repo_type="dataset"))
    os.environ.pop("HF_TOKEN")
else:
    print("PUSH_TO_HUB = False: nada foi publicado.")
''')

md("## 12. Artefatos e tempos")

code('''
OUT = WORK / "outputs"
OUT.mkdir(exist_ok=True)
shutil.copytree("results", OUT / "results", dirs_exist_ok=True)
shutil.copytree("data/labeled", OUT / "labeled", dirs_exist_ok=True)
for student in STUDENTS:
    base = MODELS / f"qwen2.5-{short(student)}-fact-extractor"
    shutil.copytree(base / "adapter", OUT / f"adapter-{short(student)}", dirs_exist_ok=True)
    shutil.copy(base / "train_log.json", OUT / f"train_log-{short(student)}.json")
    if (base / "gguf").exists():
        shutil.copytree(base / "gguf", OUT / f"gguf-{short(student)}", dirs_exist_ok=True)
(OUT / "tempos.json").write_text(json.dumps(TIMES, indent=2))
for k, v in TIMES.items():
    print(f"{k:28s} {v / 60:6.1f} min")
print("total:", round(sum(TIMES.values()) / 60, 1), "min | artefatos em", OUT)
''')


def build() -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"{OUT}: {len(CELLS)} células")
