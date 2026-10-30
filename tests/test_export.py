from pathlib import Path

from distill.export import assemble_hub_folder, gguf_commands, modelfile


def test_modelfile_uses_chatml():
    mf = modelfile("fact-extractor-0.5b-q4_k_m.gguf")
    assert mf.startswith("FROM ./fact-extractor-0.5b-q4_k_m.gguf\n")
    assert "<|im_start|>{{ .Role }}" in mf and 'PARAMETER stop "<|im_end|>"' in mf
    assert "{{{{" not in mf


def test_gguf_commands():
    cmds = gguf_commands(Path("m"), Path("o"), "x", Path("/lc"), "/py")
    assert cmds[0][:2] == ["/py", "/lc/convert_hf_to_gguf.py"] and "--outtype" in cmds[0]
    assert cmds[1] == ["/lc/build/bin/llama-quantize", "o/x-f16.gguf", "o/x-q4_k_m.gguf", "Q4_K_M"]


def test_assemble_hub_folder(tmp_path):
    merged, adapter, gg = tmp_path / "merged", tmp_path / "adapter", tmp_path / "gguf"
    for d, f in ((merged, "model.safetensors"), (adapter, "adapter_model.safetensors"), (gg, "x.gguf")):
        d.mkdir()
        (d / f).write_text("x")
    card = tmp_path / "card.md"
    card.write_text("---\nlicense: apache-2.0\n---\n")
    out = assemble_hub_folder(tmp_path / "hub", merged, adapter, gg, card)
    assert (out / "model.safetensors").exists() and (out / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "gguf" / "x.gguf").exists() and (out / "README.md").read_text().startswith("---")


def test_train_module_imports_without_gpu_libs():
    import argparse

    import distill.train as t

    args = argparse.Namespace(student="s", epochs=2, lr=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                              batch_size=8, grad_accum=2, max_length=1024, seed=42)
    cfg = t.training_config(args)
    assert cfg["lora_r"] == 16 and "q_proj" in cfg["target_modules"]


def test_legacy_tokenizer_config_converte_lista():
    from distill.export import legacy_tokenizer_config

    cfg = {"extra_special_tokens": ["<|im_start|>", "<|im_end|>"], "model_max_length": 10}
    out = legacy_tokenizer_config(cfg)
    assert "extra_special_tokens" not in out
    assert out["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"]
    assert out["model_max_length"] == 10 and "extra_special_tokens" in cfg  # não muta a entrada
    assert legacy_tokenizer_config({"extra_special_tokens": {"a": "b"}}) == {"extra_special_tokens": {"a": "b"}}


def test_conversion_view_preserva_merged(tmp_path):
    import json

    from distill.export import conversion_view

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"w")
    (merged / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": ["<x>"]}))
    view = conversion_view(merged, tmp_path / "view")
    assert (view / "model.safetensors").is_symlink()
    assert json.loads((view / "tokenizer_config.json").read_text()) == {"additional_special_tokens": ["<x>"]}
    assert json.loads((merged / "tokenizer_config.json").read_text()) == {"extra_special_tokens": ["<x>"]}
