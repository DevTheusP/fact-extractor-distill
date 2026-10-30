import json

import pytest

from distill.bench_data import bench_texts
from distill.generate import (FORBIDDEN_PAIRS, HELDOUT_AREAS, HELDOUT_RULES, Example, Generator, generate,
                              generate_splits, main, read_jsonl)
from distill.leakage import is_leak, nearest_bench


@pytest.fixture(scope="module")
def splits():
    return generate_splits(1500, 150, 200, seed=7)


def test_sizes_and_determinism(splits):
    assert [len(splits[s]) for s in ("train", "val", "test_synth")] == [1500, 150, 200]
    again = generate_splits(1500, 150, 200, seed=7)
    assert [e.to_dict() for e in again["train"]] == [e.to_dict() for e in splits["train"]]
    other = generate(50, seed=8, split="train")
    assert [e.text for e in other] != [e.text for e in splits["train"][:50]]


def test_ids_unique_and_texts_disjoint_between_splits(splits):
    ids = [e.id for s in splits.values() for e in s]
    assert len(ids) == len(set(ids))
    train = {e.text for e in splits["train"]}
    assert not train & {e.text for e in splits["val"]}
    assert not train & {e.text for e in splits["test_synth"]}


def test_all_categories_present_in_train(splits):
    cats = {e.category for e in splits["train"]}
    assert cats == set(Generator.CATEGORY_WEIGHTS)


def test_no_benchmark_leakage(splits):
    bench = set(bench_texts())
    for s in splits.values():
        for e in s:
            assert e.text not in bench
            assert not is_leak(e.text), e.text
            low = e.text.lower()
            assert "force push" not in low and "verify=false" not in low and "pagamentos" not in low


def test_forbidden_pairs_never_generated(splits):
    for s in splits.values():
        for e in s:
            for i in e.intents:
                area, _, attr = i.key.partition(".")
                assert (area.removesuffix("_service"), attr) not in FORBIDDEN_PAIRS


def test_heldout_areas_and_rules_only_in_test(splits):
    def areas(examples):
        return {i.key.split(".")[0].removesuffix("_service") for e in examples for i in e.intents}

    assert not areas(splits["train"]) & HELDOUT_AREAS
    assert areas(splits["test_synth"]) & HELDOUT_AREAS
    train_rules = {i.key for e in splits["train"] for i in e.intents}
    assert not train_rules & HELDOUT_RULES


def test_existing_keys_sorted_and_reuse_key_present(splits):
    for e in splits["train"]:
        assert e.existing_keys == sorted(set(e.existing_keys))
        if e.reuse_key:
            assert e.reuse_key in e.existing_keys
            assert any(i.key == e.reuse_key for i in e.intents)
        for d in e.distractor_keys:
            assert d in e.existing_keys and all(i.key != d for i in e.intents)


def test_category_semantics(splits):
    for e in splits["train"]:
        if e.category in ("noise", "opinion", "negated", "external"):
            assert e.intents == []
        if e.category == "multi":
            assert len(e.intents) >= 2
        if e.category == "rule":
            assert [i.kind for i in e.intents] == ["procedural"]
        if e.category == "update":
            assert e.intents[0].value in e.text


def test_text_quality(splits):
    for e in splits["train"]:
        assert "{" not in e.text and "}" not in e.text
        assert "  " not in e.text
        assert e.text[0].isupper() or e.text[0] == "["


def test_roundtrip_dict(splits):
    e = splits["train"][0]
    assert Example.from_dict(json.loads(json.dumps(e.to_dict()))) == e


def test_nearest_bench_identity():
    t = bench_texts()[0]
    assert nearest_bench(t) == (1.0, t)
    assert is_leak(t)


def test_cli_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--out", str(tmp_path), "--n-train", "30", "--n-val", "5", "--n-test", "5"])
    main()
    assert len(read_jsonl(tmp_path / "train.jsonl")) == 30
    assert len(read_jsonl(tmp_path / "test_synth.jsonl")) == 5


def test_rule_guard_marks_rule_keys_as_distractors(splits):
    rg = [e for e in splits["train"] if e.category == "rule_guard"]
    assert rg
    for e in rg:
        assert e.distractor_keys and set(e.distractor_keys) <= set(e.existing_keys)
        assert all(i.key not in e.distractor_keys for i in e.intents)
