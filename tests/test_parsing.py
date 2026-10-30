import json

from distill.parsing import Fact, normalize_key, normalize_value, parse_output, to_json

GOOD = {"facts": [{"subject_key": "billing.database_engine", "value": "PostgreSQL", "kind": "semantic",
                   "statement": "O billing usa PostgreSQL."}]}


def test_valid_output():
    p = parse_output(json.dumps(GOOD))
    assert p.json_ok and p.schema_ok
    assert p.facts == [Fact("billing.database_engine", "PostgreSQL", "semantic", "O billing usa PostgreSQL.")]


def test_empty_list_is_valid():
    p = parse_output('{"facts": []}')
    assert p.schema_ok and p.facts == []


def test_markdown_fence_is_accepted():
    p = parse_output("```json\n" + json.dumps(GOOD) + "\n```")
    assert p.schema_ok and len(p.facts) == 1


def test_invalid_json():
    p = parse_output('{"facts": [')
    assert not p.json_ok and not p.schema_ok and p.error.startswith("json")


def test_text_around_json_is_invalid():
    assert not parse_output("Aqui está: " + json.dumps(GOOD)).json_ok


def test_schema_violations():
    bad_kind = {"facts": [{**GOOD["facts"][0], "kind": "episodic"}]}
    extra = {"facts": [{**GOOD["facts"][0], "extra": 1}]}
    missing = {"facts": [{"subject_key": "a.b", "value": "x", "kind": "semantic"}]}
    for d in (bad_kind, extra, missing, {"fatos": []}, []):
        p = parse_output(json.dumps(d))
        assert p.json_ok and not p.schema_ok, d


def test_blank_key_or_value_is_dropped():
    d = {"facts": [{**GOOD["facts"][0], "value": "  "}]}
    assert parse_output(json.dumps(d)).facts == []


def test_normalization():
    assert normalize_key(" Billing.Database-Engine ") == "billing.database_engine"
    assert normalize_key("auth . provider") == "auth.provider"
    assert normalize_value(" `PostgreSQL`. ") == "postgresql"
    assert normalize_value("Ação") == "acao"


def test_to_json_is_compact_and_utf8():
    s = to_json([Fact("a.b", "ação", "semantic", "É ação.")])
    assert "ação" in s and ": " in s and json.loads(s)["facts"][0]["value"] == "ação"
    assert list(json.loads(s)["facts"][0]) == ["subject_key", "value", "kind", "statement"]
