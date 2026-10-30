import pytest
from conftest import BENCH_REPO, FakeEngine, keys_of, message_of, needs_bench

from distill.adapter import EngineLLM
from distill.prompt import SCHEMA, SYSTEM, build_user

FACT = {"subject_key": "billing.database_engine", "value": "MySQL", "kind": "semantic",
        "statement": "O billing usa MySQL."}


def test_complete_json_guided_passes_schema():
    eng = FakeEngine(lambda s, u: {"facts": [FACT]})
    llm = EngineLLM(eng, guided=True)
    assert llm.complete_json(SYSTEM, build_user("x", []), SCHEMA) == {"facts": [FACT]}
    assert eng.schemas == [SCHEMA]


def test_free_mode_sends_no_schema_and_survives_garbage():
    eng = FakeEngine(lambda s, u: "Claro! Aqui vai: {facts}")
    llm = EngineLLM(eng, guided=False)
    assert llm.complete_json(SYSTEM, build_user("x", []), SCHEMA) == {"facts": []}
    assert eng.schemas == [None] and len(llm.failures) == 1


def test_out_of_schema_counts_as_no_facts():
    llm = EngineLLM(FakeEngine(lambda s, u: {"facts": [{"subject_key": "a.b", "value": "x"}]}))
    assert llm.complete_json(SYSTEM, "u", SCHEMA) == {"facts": []}


def test_engine_error_raises():
    class Broken:
        name = "b"

        def chat(self, *a, **k):
            from distill.engines import Completion
            return Completion("", error="conexão recusada")

    with pytest.raises(RuntimeError):
        EngineLLM(Broken()).complete_json(SYSTEM, "u", SCHEMA)


def _rule_based(system, user):
    """Extrator de mentira: reconhece os fatos dos cenários do benchmark e reutiliza chaves."""
    msg, keys = message_of(user), keys_of(user)
    facts = []
    if "us-east-1" in msg or "sa-east-1" in msg:
        if msg.startswith("Contexto") or msg.startswith("Terminamos"):
            key = next((k for k in keys if "region" in k), "payments.aws_region")
            facts.append({"subject_key": key, "value": "sa-east-1" if "sa-east-1" in msg else "us-east-1",
                          "kind": "semantic", "statement": msg})
    if "pytest" in msg and "backend" in msg.lower():
        facts.append({"subject_key": "backend.test_framework", "value": "pytest", "kind": "semantic", "statement": msg})
    if "vitest" in msg:
        facts.append({"subject_key": "frontend.test_framework", "value": "vitest", "kind": "semantic", "statement": msg})
    if "force push na main" in msg and msg.startswith("Regra"):
        facts.append({"subject_key": "git.main_force_push", "value": "forbidden", "kind": "procedural", "statement": msg})
    return {"facts": facts}


@needs_bench
def test_adapter_runs_benchmark_layer1_with_fake_llm(tmp_path):
    pytest.importorskip("sqlite_vec")
    from distill.bench import headline, run_layer1

    import importlib
    import sys
    sys.path.insert(0, str(BENCH_REPO))
    HashEmbedder = importlib.import_module("memory.providers").HashEmbedder

    llm = EngineLLM(FakeEngine(_rule_based))
    summary = run_layer1(llm, HashEmbedder(), BENCH_REPO, tmp_path, "fake", seeds=[1], noise_levels=[3],
                         gap_days=[0])
    h = headline(summary)
    assert {"v0_no_memory", "v1_naive", "v2_keyed"} <= set(h)
    assert llm.calls > 0 and not llm.failures
    # com o extrator certo, a v2 recupera a regra rara e não deixa a injeção entrar
    assert h["v2_keyed"]["ret_injection"] == 1.0
    assert (tmp_path / "summary_fake.csv").exists() and (tmp_path / "extract_cache_fake.json").exists()
