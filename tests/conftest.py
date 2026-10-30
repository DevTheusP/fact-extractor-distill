import json
import os
from pathlib import Path

import pytest

from distill.engines import Completion

ROOT = Path(__file__).resolve().parent.parent
BENCH_REPO = Path(os.environ.get("BENCH_REPO", ROOT.parent / "agent-memory-bench"))


def bench_available() -> bool:
    return (BENCH_REPO / "eval" / "run.py").exists()


needs_bench = pytest.mark.skipif(not bench_available(), reason="agent-memory-bench não está ao lado deste repo")


class FakeEngine:
    """Responde com uma função de (system, user) -> dict|str. Registra se pediu schema."""

    def __init__(self, fn, name="fake"):
        self.fn, self.name, self.calls, self.schemas = fn, name, 0, []

    def chat(self, messages, schema=None, max_tokens=512):
        self.calls += 1
        self.schemas.append(schema)
        out = self.fn(messages[0]["content"], messages[-1]["content"])
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return Completion(text, completion_tokens=len(text) // 4, prompt_tokens=100, latency_s=0.01)


def message_of(user: str) -> str:
    return user.split("Mensagem:\n", 1)[1]


def keys_of(user: str) -> list[str]:
    block = user.split("Chaves existentes:\n", 1)[1].split("\n\nMensagem:", 1)[0]
    return [line[2:] for line in block.splitlines() if line.startswith("- ")]


@pytest.fixture
def fake_engine_factory():
    return FakeEngine
