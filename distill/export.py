"""Exportação: modelo merged, GGUF q4 para Ollama/llama.cpp e push opcional para o Hub.

Uso:
    python -m distill.export merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/x/adapter --out models/x/merged
    python -m distill.export gguf --merged models/x/merged --out models/x/gguf --name fact-extractor-0.5b \\
        --llama-cpp /kaggle/tmp/llama.cpp --python /kaggle/tmp/gguf-env/bin/python
    python -m distill.export push --folder models/x/hub --repo usuario/repo   # token vem do ambiente (HF_TOKEN)

O token do Hugging Face nunca passa por aqui como argumento nem fica em
arquivo: o huggingface_hub lê HF_TOKEN do ambiente, que o notebook preenche a
partir dos Kaggle Secrets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

# Template ChatML do Qwen2.5 para o Ollama, sem ferramentas. O benchmark sempre
# manda mensagem de sistema, então não há sistema padrão aqui.
MODELFILE = '''FROM ./{gguf}
TEMPLATE """{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
PARAMETER num_ctx 2048
'''


def modelfile(gguf_name: str) -> str:
    return MODELFILE.format(gguf=gguf_name)


def merge(base: str, adapter: Path, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(out))


def gguf_commands(merged: Path, out: Path, name: str, llama_cpp: Path, python: str,
                  quant: str = "Q4_K_M") -> list[list[str]]:
    f16 = out / f"{name}-f16.gguf"
    q = out / f"{name}-{quant.lower()}.gguf"
    return [
        [python, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"],
        [str(llama_cpp / "build" / "bin" / "llama-quantize"), str(f16), str(q), quant],
    ]


def legacy_tokenizer_config(cfg: dict) -> dict:
    """tokenizer_config.json do transformers 5 lido pelo transformers 4 do conversor do llama.cpp.

    O transformers 5 grava os tokens especiais extras como lista em
    `extra_special_tokens`; o 4 espera ali um dicionário e quebra com
    "'list' object has no attribute 'keys'". A lista volta para o nome antigo,
    `additional_special_tokens`.
    """
    cfg = dict(cfg)
    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        del cfg["extra_special_tokens"]
        merged = list(cfg.get("additional_special_tokens") or [])
        cfg["additional_special_tokens"] = merged + [t for t in extra if t not in merged]
    return cfg


def conversion_view(merged: Path, view: Path) -> Path:
    """Cópia leve do modelo merged para o conversor: links simbólicos, tokenizer_config ajustado.

    O merged original fica intacto, no formato do transformers 5, para quem
    baixar do Hub.
    """
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    for f in merged.iterdir():
        if f.name == "tokenizer_config.json":
            cfg = json.loads(f.read_text())
            (view / f.name).write_text(json.dumps(legacy_tokenizer_config(cfg), ensure_ascii=False, indent=2))
        else:
            (view / f.name).symlink_to(f.resolve())
    return view


def gguf(merged: Path, out: Path, name: str, llama_cpp: Path, python: str, quant: str = "Q4_K_M") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    view = conversion_view(merged, out / "_convert_src")
    try:
        for cmd in gguf_commands(view, out, name, llama_cpp, python, quant):
            cmd = [str(c) for c in cmd]
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(view, ignore_errors=True)  # não deixa links soltos irem para o Hub
    q = out / f"{name}-{quant.lower()}.gguf"
    (out / f"{name}-f16.gguf").unlink(missing_ok=True)  # só o quantizado vai para o Hub
    (out / "Modelfile").write_text(modelfile(q.name))
    return q


def push(folder: Path, repo: str, repo_type: str = "model", private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi()  # lê HF_TOKEN do ambiente
    api.create_repo(repo, repo_type=repo_type, private=private, exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=repo_type,
                             commit_message="Envia artefatos do fact-extractor-distill")
    return str(info)


def assemble_hub_folder(out: Path, merged: Path, adapter: Path, gguf_dir: Path | None, card: Path,
                        extra: dict[str, Path] | None = None) -> Path:
    """Monta a pasta do repositório do modelo: merged na raiz, adapter/ e gguf/ em subpastas."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(merged, out)
    shutil.copytree(adapter, out / "adapter")
    if gguf_dir and gguf_dir.exists():
        shutil.copytree(gguf_dir, out / "gguf")
    shutil.copy(card, out / "README.md")
    for name, path in (extra or {}).items():
        shutil.copy(path, out / name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--adapter", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gguf")
    g.add_argument("--merged", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--llama-cpp", type=Path, required=True)
    g.add_argument("--python", default="python")
    g.add_argument("--quant", default="Q4_K_M")
    p = sub.add_parser("push")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--private", action="store_true")
    args = ap.parse_args()
    if args.cmd == "merge":
        merge(args.base, args.adapter, args.out)
    elif args.cmd == "gguf":
        print(gguf(args.merged, args.out, args.name, args.llama_cpp, args.python, args.quant))
    else:
        print(push(args.folder, args.repo, args.repo_type, args.private))


if __name__ == "__main__":
    main()
