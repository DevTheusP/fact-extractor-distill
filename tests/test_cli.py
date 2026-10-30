"""As CLIs que o notebook chama, contra um servidor HTTP falso (no lugar do vLLM)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from distill import evaluate as ev
from distill import generate as gen
from distill import label as lb
from distill import report as rp


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msg = body["messages"][-1]["content"].split("Mensagem:\n", 1)[1]
            facts = []
            if "usa" in msg:
                facts = [{"subject_key": "x.y", "value": msg.split()[-1].rstrip("."), "kind": "semantic",
                          "statement": msg}]
            data = json.dumps({"choices": [{"message": {"content": json.dumps({"facts": facts})}}],
                               "usage": {"completion_tokens": 20}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_generate_label_evaluate_report(server, tmp_path, monkeypatch):
    g, lab, out = tmp_path / "generated", tmp_path / "labeled", tmp_path / "eval"
    monkeypatch.setattr("sys.argv", ["x", "--out", str(g), "--n-train", "20", "--n-val", "8", "--n-test", "10"])
    gen.main()
    monkeypatch.setattr("sys.argv", ["x", "--base-url", server, "--data-dir", str(g), "--out-dir", str(lab),
                                     "--bench", "--workers", "4"])
    lb.main()
    rep = json.loads((lab / "label_report.json").read_text())
    assert set(rep) == {"train", "val", "test_synth", "test_bench"}
    assert rep["train"]["sft"]["kept"] == 20
    assert len(gen.read_jsonl(lab / "sft_val.jsonl")) == 8

    monkeypatch.setattr("sys.argv", ["x", "--base-url", server, "--model", "student", "--name", "aluno-base",
                                     "--labeled-dir", str(lab), "--out", str(out), "--workers", "4",
                                     "--latency-sample", "3"])
    ev.main()
    reports = json.loads((out / "report_aluno-base.json").read_text())
    # 2 conjuntos x 2 modos; test_bench tem 3 referências, test_synth 2
    assert len(reports) == 2 * 2 + 2 * 3
    same = [r for r in reports if r["dataset"] == "test_synth|ref=professor"]
    assert all(r["overall"]["strict_f1"] in (1.0, None) for r in same)

    monkeypatch.setattr("sys.argv", ["x", "--eval-dir", str(out), "--bench-dir", str(tmp_path / "nada"),
                                     "--out", str(tmp_path / "res"), "--no-plots"])
    rp.main()
    assert "test_bench|ref=professor_ollama" in (tmp_path / "res" / "RESULTS.md").read_text()
