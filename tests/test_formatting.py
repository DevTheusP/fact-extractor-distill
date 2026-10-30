import json

from distill.formatting import render_chatml, render_record, sft_record
from distill.prompt import SYSTEM


def test_sft_record_prompt_completion():
    facts = [{"subject_key": "a.b", "value": "x", "kind": "semantic", "statement": "A é x."}]
    rec = sft_record("A é x.", ["c.d"], facts, "train-1")
    assert rec["id"] == "train-1"
    assert [m["role"] for m in rec["prompt"]] == ["system", "user"]
    assert rec["prompt"][0]["content"] == SYSTEM
    assert rec["completion"][0]["role"] == "assistant"
    assert json.loads(rec["completion"][0]["content"]) == {"facts": facts}


def test_empty_label_trains_empty_list():
    rec = sft_record("Hoje foi corrido.", [], [])
    assert rec["completion"][0]["content"] == '{"facts": []}'


def test_render_chatml_matches_qwen_template():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert render_chatml(msgs, add_generation_prompt=True) == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")


def test_default_system_when_missing():
    assert render_chatml([{"role": "user", "content": "U"}]).startswith(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")


def test_render_record_ends_with_assistant_turn():
    rec = sft_record("t", [], [])
    text = render_record(rec)
    assert text.endswith('<|im_start|>assistant\n{"facts": []}<|im_end|>\n')
    assert text.count("<|im_start|>") == 3
