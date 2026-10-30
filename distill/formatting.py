"""Formatação para SFT no chat template do Qwen2.5 (ChatML).

O dataset de treino fica no formato conversacional prompt-completion do TRL:
    {"prompt": [system, user], "completion": [assistant]}
Assim o SFTTrainer aplica o chat template do próprio tokenizer e calcula a
perda só nos tokens da resposta (completion_only_loss, padrão do TRL para esse
formato), sem depender das tags {% generation %} que assistant_only_loss exige.

render_chatml reproduz o template do Qwen2.5 em Python puro. Serve para os
testes locais (sem baixar tokenizer) e o notebook confere que ele bate com
tokenizer.apply_chat_template antes de treinar.
"""

from __future__ import annotations

from .parsing import to_json
from .prompt import build_messages


def sft_record(text: str, existing_keys: list[str], facts: list[dict], example_id: str | None = None) -> dict:
    rec = {
        "prompt": build_messages(text, existing_keys),
        "completion": [{"role": "assistant", "content": to_json(facts)}],
    }
    if example_id is not None:
        rec["id"] = example_id
    return rec


def render_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Igual ao chat template do Qwen2.5-Instruct quando há mensagem de sistema e sem ferramentas."""
    out = []
    if not messages or messages[0]["role"] != "system":
        out.append("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n")
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def render_record(rec: dict) -> str:
    """Texto completo de treino (prompt + resposta)."""
    return render_chatml(rec["prompt"] + rec["completion"])
