"""Sobe e derruba o `vllm serve` como subprocesso (o vLLM fica num venv separado).

O vLLM fixa a própria versão do torch; instalado no ambiente principal do
Kaggle, ele trocaria o torch que o treino usa. Por isso ele roda isolado em
/kaggle/tmp/vllm-env e conversa com o resto só por HTTP.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def serve_command(vllm_bin: str, model: str, *, served_name: str, port: int = 8000, dtype: str = "half",
                  max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                  enforce_eager: bool = False, extra: list[str] | None = None) -> list[str]:
    """Linha de comando do `vllm serve` (CLI documentada do vLLM)."""
    cmd = [
        str(vllm_bin), "serve", str(model),
        "--served-model-name", served_name,
        "--port", str(port),
        "--dtype", dtype,  # T4 (Turing) não tem bfloat16
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--seed", "0",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    return cmd + [str(a) for a in (extra or [])]


def wait_ready(base_url: str, timeout: float = 900.0, proc: subprocess.Popen | None = None,
               log: Path | None = None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            tail = log.read_text(errors="replace")[-4000:] if log and log.exists() else ""
            raise RuntimeError(f"vllm serve saiu com código {proc.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
                json.loads(r.read())
                return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"servidor não respondeu em {timeout:.0f}s")


class VLLMServer:
    """with VLLMServer(...) as base_url: ... — derruba o processo e libera a GPU na saída."""

    def __init__(self, vllm_bin: str, model: str, served_name: str, *, port: int = 8000, gpu: str = "0",
                 log_dir: Path = Path("logs"), **kw):
        self.cmd = serve_command(vllm_bin, model, served_name=served_name, port=port, **kw)
        self.base_url = f"http://127.0.0.1:{port}"
        self.gpu = gpu
        self.log = Path(log_dir) / f"vllm_{served_name}.log"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> str:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu}
        self._fh = self.log.open("w")
        print("subindo:", " ".join(self.cmd), flush=True)
        self.proc = subprocess.Popen(self.cmd, stdout=self._fh, stderr=subprocess.STDOUT, env=env)
        wait_ready(self.base_url, proc=self.proc, log=self.log)
        print(f"pronto em {self.base_url} (log em {self.log})", flush=True)
        return self.base_url

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._fh.close()
