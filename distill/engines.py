"""Motores de inferência com uma interface só: chat(messages, schema) -> Completion.

- OpenAIChatEngine: qualquer servidor compatível com a API da OpenAI. No Kaggle
  é o `vllm serve`; localmente pode ser o llama.cpp server ou o endpoint
  /v1 do Ollama. Com schema, pede saída guiada (response_format json_schema),
  o equivalente ao `format` do Ollama que o benchmark usa. Só biblioteca padrão.
- HFEngine: transformers puro, plano B se o vLLM não subir na GPU. Não tem
  gramática: o schema é ignorado e a saída é livre.

Nenhum motor lê token de lugar nenhum: o servidor vLLM local não exige chave.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    latency_s: float | None = None
    error: str | None = None


class Engine(Protocol):
    name: str

    def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 512) -> Completion: ...


class OpenAIChatEngine:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0, name: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = name or model

    def payload(self, messages: list[dict], schema: dict | None, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "facts", "schema": schema, "strict": True},
            }
        return body

    def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 512) -> Completion:
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(self.payload(messages, schema, max_tokens)).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            return Completion("", latency_s=time.perf_counter() - t0, error=repr(e))
        dt = time.perf_counter() - t0
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"].get("content") or ""
        return Completion(text, usage.get("completion_tokens"), usage.get("prompt_tokens"), dt)


class HFEngine:
    """transformers + generate guloso. Carrega na primeira chamada."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "float16",
                 load_in_4bit: bool = False, name: str | None = None):
        self.model_path, self.device, self.dtype, self.load_in_4bit = model_path, device, dtype, load_in_4bit
        self.name = name or model_path
        self._model = self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw: dict = {"dtype": getattr(torch, self.dtype)}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
            kw["device_map"] = "auto"
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._tok.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kw)
        if not self.load_in_4bit:
            self._model.to(self.device)
        self._model.eval()

    def chat_batch(self, batch: list[list[dict]], schema: dict | None = None, max_tokens: int = 512) -> list[Completion]:
        import torch

        self._load()
        texts = [self._tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = self._tok(texts, return_tensors="pt", padding=True).to(self._model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                       pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        dt = time.perf_counter() - t0
        gen = out[:, enc["input_ids"].shape[1]:]
        res = []
        for row in gen:
            ids = [int(t) for t in row if int(t) != (self._tok.pad_token_id or -1)]
            text = self._tok.decode(ids, skip_special_tokens=True)
            res.append(Completion(text, len(ids), None, dt / len(batch) if len(batch) > 1 else dt))
        return res

    def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 512) -> Completion:
        return self.chat_batch([messages], schema, max_tokens)[0]


def run_many(engine: Engine, batch: list[list[dict]], schema: dict | None = None, max_tokens: int = 512,
             workers: int = 32, batch_size: int = 16, progress: bool = True) -> list[Completion]:
    """Roda vários pedidos mantendo a ordem. Servidor: threads concorrentes (o vLLM agrupa sozinho)."""
    out: list[Completion] = []
    total = len(batch)
    t0 = time.perf_counter()

    def report(done: int) -> None:
        if progress and (done == total or done % max(1, total // 10) == 0):
            print(f"  {done}/{total} ({time.perf_counter() - t0:.0f}s)", flush=True)

    if workers <= 1:  # um pedido por vez: é o que mede latência
        for m in batch:
            out.append(engine.chat(m, schema, max_tokens))
            report(len(out))
        return out
    if hasattr(engine, "chat_batch"):
        for i in range(0, total, batch_size):
            out += engine.chat_batch(batch[i:i + batch_size], schema, max_tokens)
            report(len(out))
        return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c in pool.map(lambda m: engine.chat(m, schema, max_tokens), batch):
            out.append(c)
            report(len(out))
    return out
