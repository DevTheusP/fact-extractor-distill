"""Fine-tuning LoRA do aluno com TRL + PEFT (roda no Kaggle, em GPU).

Por que TRL/PEFT e não Unsloth: o aluno tem 0,5B parâmetros e cabe com folga
numa T4 em LoRA comum, então o ganho de velocidade do Unsloth não paga o
risco de ele remendar versões do transformers no ambiente do Kaggle. TRL e
PEFT são as bibliotecas de referência da Hugging Face e o adaptador sai num
formato que qualquer um carrega.

Precisão: a T4 (Turing) não tem bfloat16. O modelo base carrega em float32
e o treino usa autocast fp16 (fp16=True), com os pesos do LoRA em float32.

Uso:
    python -m distill.train --student Qwen/Qwen2.5-0.5B-Instruct --data-dir data/labeled \\
        --out models/qwen2.5-0.5b-fact-extractor-lora --epochs 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Módulos lineares do bloco Qwen2 (atenção + MLP).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def training_config(args) -> dict:
    """Hiperparâmetros num dict simples (também vai para o model card)."""
    return {
        "student": args.student,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": TARGET_MODULES,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_length": args.max_length,
        "warmup_ratio": 0.05,
        "lr_scheduler": "cosine",
        "seed": args.seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data-dir", type=Path, default=Path("data/labeled"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1, help="corta o treino (smoke test)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = training_config(args)
    files = {"train": str(args.data_dir / "sft_train.jsonl"), "validation": str(args.data_dir / "sft_val.jsonl")}
    ds = load_dataset("json", data_files=files)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])

    tok = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16 if bf16 else torch.float32)

    sft_args = SFTConfig(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # no transformers 5, valor < 1 é fração do total (o antigo warmup_ratio)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        max_length=args.max_length,
        completion_only_loss=True,  # perda só no JSON do professor
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tok, peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    final_eval = trainer.evaluate()

    adapter_dir = args.out / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    log = {"config": cfg, "bf16": bf16, "train_runtime_s": result.metrics.get("train_runtime"),
           "train_loss": result.training_loss, "final_eval": final_eval,
           "n_train": len(ds["train"]), "n_val": len(ds["validation"]),
           "log_history": trainer.state.log_history}
    (args.out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "log_history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
