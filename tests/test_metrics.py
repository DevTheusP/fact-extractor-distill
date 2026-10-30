import json

import pytest

from distill.metrics import Report, aggregate, by_category, markdown_table, score_case
from distill.parsing import parse_output


def F(key, value, kind="semantic"):
    return {"subject_key": key, "value": value, "kind": kind, "statement": "s"}


def P(*facts):
    return parse_output(json.dumps({"facts": list(facts)}))


def test_perfect_match():
    s = score_case("1", "c", [], [F("a.b", "x")], P(F("a.b", "x")))
    assert (s.strict_tp, s.value_tp, s.kind_ok) == (1, 1, 1)
    m = aggregate([s])
    assert m["strict_f1"] == m["value_f1"] == 1.0


def test_key_mismatch_counts_only_by_value():
    s = score_case("1", "c", [], [F("billing.database_engine", "PostgreSQL")], P(F("billing.db", "postgresql")))
    assert s.strict_tp == 0 and s.value_tp == 1


def test_normalized_key_match():
    s = score_case("1", "c", [], [F("billing.database_engine", "X")], P(F("Billing.Database-Engine", "x")))
    assert s.strict_tp == 1


def test_value_containment():
    s = score_case("1", "c", [], [F("a.b", "Kubernetes (EKS)")], P(F("a.c", "Kubernetes")))
    assert s.value_tp == 1
    s = score_case("1", "c", [], [F("a.b", "Go")], P(F("a.c", "Google")))
    assert s.value_tp == 0  # valor curto demais para casar por inclusão


def test_one_to_one_pairing():
    s = score_case("1", "c", [], [F("a.b", "x")], P(F("a.b", "x"), F("a.b", "x")))
    assert s.strict_tp == 1 and s.n_pred == 2
    m = aggregate([s])
    assert m["strict_precision"] == 0.5 and m["strict_recall"] == 1.0


def test_invalid_output_counts_as_empty():
    s = score_case("1", "c", [], [F("a.b", "x")], parse_output("não é json"))
    assert not s.json_ok and s.n_pred == 0 and not s.pred_empty
    m = aggregate([s])
    assert m["json_valid"] == 0 and m["strict_recall"] == 0 and m["false_empty_rate"] == 0


def test_kind_accuracy():
    s = score_case("1", "c", [], [F("a.b", "x", "procedural")], P(F("a.b", "x", "semantic")))
    assert s.kind_pairs == 1 and s.kind_ok == 0


def test_reuse_accuracy_and_false_reuse():
    existing = ["billing.db_engine", "search.db_engine"]
    ok = score_case("1", "c", existing, [F("billing.db_engine", "MySQL")], P(F("billing.db_engine", "MySQL")))
    wrong = score_case("2", "c", existing, [F("billing.db_engine", "MySQL")], P(F("billing.database", "MySQL")))
    false = score_case("3", "c", existing, [F("auth.db_engine", "MySQL")], P(F("search.db_engine", "MySQL")))
    m = aggregate([ok, wrong, false])
    assert m["reuse_accuracy"] == 0.5
    assert m["false_reuse_rate"] == 0.5  # 2 previsões com chave existente, 1 indevida


def test_empty_accuracy():
    good = score_case("1", "noise", [], [], P())
    bad = score_case("2", "noise", [], [], P(F("a.b", "x")))
    miss = score_case("3", "config", [], [F("a.b", "x")], P())
    m = aggregate([good, bad, miss])
    assert m["empty_accuracy"] == 0.5
    assert m["false_empty_rate"] == 1.0
    assert set(by_category([good, bad, miss])) == {"noise", "config"}


def test_not_applicable_is_none():
    m = aggregate([score_case("1", "c", [], [], P())])
    assert m["strict_f1"] is None and m["reuse_accuracy"] is None and m["empty_accuracy"] == 1.0


def test_latency_aggregation():
    scores = [score_case(str(i), "c", [], [], P(), latency_s=t, completion_tokens=10)
              for i, t in enumerate([0.1, 0.2, 0.3, 0.4])]
    m = aggregate(scores)
    assert m["latency_mean_s"] == pytest.approx(0.25)
    assert m["tokens_per_s"] == pytest.approx(40 / 1.0)
    assert m["latency_p95_s"] == 0.4


def test_markdown_table():
    r = Report("aluno", "test", "guided", {"schema_valid": 1.0, "strict_f1": 0.5, "latency_p50_s": 0.123,
                                            "tokens_per_s": 250.4})
    t = markdown_table([r])
    assert "| aluno | test | guided | 100.0% | 50.0% |" in t
    assert "0.12" in t and "250" in t and "| - |" in t
