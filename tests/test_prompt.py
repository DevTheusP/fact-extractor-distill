from conftest import BENCH_REPO, needs_bench

from distill.prompt import SCHEMA, SYSTEM, build_messages, build_user


def test_user_message_matches_benchmark_format():
    assert build_user("oi", []) == "Chaves existentes:\n(nenhuma ainda)\n\nMensagem:\noi"
    assert build_user("oi", ["a.b", "c.d"]) == "Chaves existentes:\n- a.b\n- c.d\n\nMensagem:\noi"


def test_messages_have_system_then_user():
    msgs = build_messages("texto", ["x.y"])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM


@needs_bench
def test_system_and_schema_are_identical_to_the_benchmark():
    import importlib.util
    import sys
    import types

    # carrega memory/extract.py do benchmark sem importar o pacote inteiro (que exige sqlite-vec)
    pkg = types.ModuleType("memory")
    pkg.__path__ = [str(BENCH_REPO / "memory")]
    saved = sys.modules.get("memory")
    sys.modules["memory"] = pkg
    try:
        for name in ("providers", "extract"):
            spec = importlib.util.spec_from_file_location(f"memory.{name}", BENCH_REPO / "memory" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"memory.{name}"] = mod
            spec.loader.exec_module(mod)
        bench = sys.modules["memory.extract"]
        assert bench.SYSTEM == SYSTEM
        assert bench.SCHEMA == SCHEMA

        class Spy:
            def complete_json(self, system, user, schema):
                self.user = user
                return {"facts": []}

        spy = Spy()
        bench.LLMExtractor(spy).extract("mensagem qualquer", ["a.b"])
        assert spy.user == build_user("mensagem qualquer", ["a.b"])
    finally:
        for k in ("memory", "memory.providers", "memory.extract"):
            sys.modules.pop(k, None)
        if saved is not None:
            sys.modules["memory"] = saved
