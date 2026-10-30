"""Pipeline local de ponta a ponta com motores falsos: gerar -> rotular -> SFT -> avaliar -> relatório."""

import json

import pytest
from conftest import FakeEngine, keys_of, message_of

from distill.bench_data import bench_test_cases, ollama_cache, ollama_labels
from distill.evaluate import evaluate, score
from distill.generate import Example, generate_splits, read_jsonl, write_jsonl
from distill.label import build_sft, gold_reference, label_examples, teacher_vs_gold
from distill.prompt import SCHEMA
from distill.report import write_results


def oracle_for(examples):
    """Professor de mentira que responde a intenção do template (o gabarito)."""
    table = {(e.text, tuple(e.existing_keys)): e for e in examples}

    def fn(system, user):
        e = table[(message_of(user), tuple(keys_of(user)))]
        return {"facts": [{"subject_key": i.key, "value": i.value, "kind": i.kind, "statement": e.text}
                          for i in e.intents if not i.optional]}
    return fn


@pytest.fixture(scope="module")
def data():
    s = generate_splits(80, 10, 20, seed=3)
    s["test_bench"] = bench_test_cases()
    return s


def test_label_and_build_sft(data):
    teacher = FakeEngine(oracle_for(data["train"]), name="teacher")
    rows = label_examples(teacher, data["train"], workers=4)
    assert len(rows) == 80 and all(r["teacher"]["schema_ok"] for r in rows)
    assert teacher.schemas[0] == SCHEMA  # rotulagem guiada
    records, stats = build_sft(rows)
    assert stats == {"kept": 80, "dropped_invalid": 0}
    assert records[0]["id"] == rows[0]["id"]
    q = teacher_vs_gold(rows)
    assert q["overall"]["value_f1"] == 1.0 and q["overall"]["empty_accuracy"] == 1.0


def test_invalid_teacher_labels_are_dropped(data):
    rows = label_examples(FakeEngine(lambda s, u: "lixo"), data["train"][:5], workers=1)
    records, stats = build_sft(rows)
    assert records == [] and stats["dropped_invalid"] == 5


def test_gold_reference_skips_optional():
    row = {"intents": [{"key": "a.b", "value": "x", "kind": "semantic", "optional": False},
                       {"key": "a.c", "value": "y", "kind": "semantic", "optional": True}]}
    ref, optional = gold_reference(row)
    assert [r["subject_key"] for r in ref] == ["a.b"] and optional == ["y"]


def test_bench_cases_come_from_published_cache():
    cases = bench_test_cases()
    assert len(cases) == len(ollama_cache()) == 131
    assert {c.category for c in cases} == {"bench_noise", "bench_update", "bench_scope", "bench_rare_fact"}
    # a injeção (fonte web) nunca chegou ao extrator no benchmark, então não está no teste
    assert not any("verify=False" in c.text for c in cases)
    update4 = [c for c in cases if c.text.startswith("Terminamos a migração")]
    assert update4 and update4[0].reuse_key == "payment_service.region"
    assert len(ollama_labels()) == 131


def test_evaluate_end_to_end(data, tmp_path):
    labeled = tmp_path / "labeled"
    all_examples = [e for s in data.values() for e in s]
    teacher = FakeEngine(oracle_for(all_examples), name="teacher")
    for split in ("test_synth", "test_bench"):
        write_jsonl(labeled / f"{split}.jsonl", label_examples(teacher, data[split], workers=4))

    # aluno perfeito (copia o professor) e aluno que sempre devolve lista vazia
    good = FakeEngine(oracle_for(all_examples), name="aluno-destilado")
    lazy = FakeEngine(lambda s, u: {"facts": []}, name="aluno-base")
    out = tmp_path / "eval"
    rg = evaluate(good, "aluno-destilado", labeled, ["test_synth", "test_bench"], ["guided", "free"], out,
                  workers=4, latency_sample=5)
    rl = evaluate(lazy, "aluno-base", labeled, ["test_synth", "test_bench"], ["guided"], out,
                  workers=4, latency_sample=5)

    g = {(r.dataset, r.mode): r.overall for r in rg}
    assert g[("test_synth|ref=professor", "guided")]["strict_f1"] == 1.0
    assert g[("test_synth|ref=gabarito", "free")]["value_f1"] == 1.0
    assert g[("test_bench|ref=professor", "guided")]["reuse_accuracy"] == 1.0
    assert "latency_p50_s" in g[("test_synth|ref=professor", "guided")]
    assert ("test_bench|ref=professor_ollama", "guided") in g
    lz = {(r.dataset, r.mode): r.overall for r in rl}
    assert lz[("test_synth|ref=professor", "guided")]["strict_recall"] == 0.0
    assert lz[("test_bench|ref=professor", "guided")]["empty_accuracy"] == 1.0

    preds = read_jsonl(out / "pred_aluno-base_test_bench_guided.jsonl")
    assert len(preds) == 131 and sum(p["latency_s"] is not None for p in preds) == 5

    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    for name, v2 in (("aluno-destilado", 0.97), ("professor", 0.98)):
        h = {"v1_naive": {"retrieval": 0.62, **{f"ret_{s}": 0.5 for s in ("update", "scope", "rare_fact", "injection")}},
             "v2_keyed": {"retrieval": v2, **{f"ret_{s}": 1.0 for s in ("update", "scope", "rare_fact", "injection")}}}
        (bench_dir / f"headline_{name}.json").write_text(json.dumps(
            {"name": name, "guided": True, "extractor_calls": 10, "extractor_failures": 0, "headline": h}))
    pytest.importorskip("matplotlib")
    text = write_results(out, bench_dir, tmp_path / "results")
    assert "aluno-destilado" in text and "camada 1" in text and "97.0%" in text
    assert (tmp_path / "results" / "benchmark_recuperacao.png").exists()
    assert (tmp_path / "results" / "latencia.png").exists()


def test_score_against_ollama_reference():
    rows = [{**e.to_dict(), "teacher": {"facts": []}} for e in bench_test_cases()]
    labels = ollama_labels()
    preds = [{"raw": json.dumps({"facts": labels[(r["text"], tuple(r["existing_keys"]))]}, ensure_ascii=False),
              "latency_s": None, "completion_tokens": None} for r in rows]
    overall, _ = score(rows, preds, "professor_ollama")
    assert overall["strict_f1"] == 1.0 and overall["empty_accuracy"] == 1.0
