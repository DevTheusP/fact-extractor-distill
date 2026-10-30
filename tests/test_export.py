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
