"""Adaptadores para a interface do agent-memory-bench (memory/providers.py).

O benchmark espera:
    class LLM(Protocol):
        def complete_json(self, system: str, user: str, schema: dict) -> dict: ...
    class Embedder(Protocol):
        dim: int
        def embed(self, texts: list[str]) -> list[list[float]]: ...

EngineLLM coloca qualquer motor de distill.engines (o aluno destilado servido
pelo vLLM, por exemplo) no lugar do OllamaLLM, sem tocar no código do
benchmark: o LLMExtractor continua montando o prompt e lendo o JSON.

SentenceTransformerEmbedder substitui o OllamaEmbedder (bge-m3 via Ollama),
que não existe no Kaggle, pelo mesmo modelo BAAI/bge-m3 via sentence-transformers.
"""

from __future__ import annotations

import math

from .engines import Completion, Engine
from .parsing import parse_output


class EngineLLM:
    """complete_json sobre um Engine.

    guided=True pede saída guiada pelo schema, como o `format` do Ollama no
    benchmark original. guided=False mede o modelo sem gramática; resposta
    fora do schema conta como "nenhum fato" (o benchmark original trataria
    JSON inválido como exceção e derrubaria a rodada) e fica em failures.
    """

    def __init__(self, engine: Engine, guided: bool = True, max_tokens: int = 512):
        self.engine = engine
        self.guided = guided
        self.max_tokens = max_tokens
        self.calls = 0
        self.failures: list[dict] = []
        self.completions: list[Completion] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls += 1
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        c = self.engine.chat(messages, schema if self.guided else None, self.max_tokens)
        self.completions.append(c)
        if c.error:
            raise RuntimeError(f"motor falhou: {c.error}")
        parsed = parse_output(c.text)
        if not parsed.schema_ok:
            # mesma regra das métricas: fora do schema conta como nenhum fato
            self.failures.append({"user": user, "text": c.text, "error": parsed.error})
            return {"facts": []}
        return {"facts": [f.to_dict() for f in parsed.facts]}


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-m3", device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return [_unit([float(x) for x in v]) for v in vecs]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]
