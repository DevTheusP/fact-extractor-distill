import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from distill.engines import Completion, OpenAIChatEngine, run_many
from distill.prompt import SCHEMA
from distill.server import serve_command


@pytest.fixture
def fake_server():
    """Servidor HTTP local que imita /v1/chat/completions do vLLM."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            user = body["messages"][-1]["content"]
            content = '{"facts": []}' if "ruído" in user else "não é json"
            out = {"choices": [{"message": {"role": "assistant", "content": content}}],
                   "usage": {"prompt_tokens": 50, "completion_tokens": 7}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}", received
    srv.shutdown()


def test_guided_request_payload(fake_server):
    url, received = fake_server
    eng = OpenAIChatEngine(url, "student")
    c = eng.chat([{"role": "user", "content": "ruído"}], SCHEMA, max_tokens=64)
    assert c.text == '{"facts": []}' and c.completion_tokens == 7 and c.latency_s > 0
    body = received[-1]
    assert body["model"] == "student" and body["temperature"] == 0 and body["max_tokens"] == 64
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_free_request_has_no_response_format(fake_server):
    url, received = fake_server
    OpenAIChatEngine(url, "m").chat([{"role": "user", "content": "x"}])
    assert "response_format" not in received[-1]


def test_connection_error_is_reported_not_raised():
    c = OpenAIChatEngine("http://127.0.0.1:9", "m", timeout=2).chat([{"role": "user", "content": "x"}])
    assert c.error and c.text == ""


def test_run_many_keeps_order(fake_server):
    url, _ = fake_server
    eng = OpenAIChatEngine(url, "m")
    batch = [[{"role": "user", "content": "ruído" if i % 2 else "fato"}] for i in range(20)]
    out = run_many(eng, batch, workers=8, progress=False)
    assert [c.text == '{"facts": []}' for c in out] == [bool(i % 2) for i in range(20)]
    seq = run_many(eng, batch[:3], workers=1, progress=False)
    assert len(seq) == 3


def test_run_many_prefers_batches_but_not_for_latency():
    class Batchy:
        name = "b"
        batches, singles = [], 0

        def chat_batch(self, b, schema=None, max_tokens=512):
            self.batches.append(len(b))
            return [Completion("x") for _ in b]

        def chat(self, m, schema=None, max_tokens=512):
            self.singles += 1
            return Completion("x")

    e = Batchy()
    run_many(e, [[]] * 10, batch_size=4, progress=False)
    assert e.batches == [4, 4, 2]
    run_many(e, [[]] * 3, workers=1, progress=False)
    assert e.singles == 3


def test_serve_command():
    cmd = serve_command("/env/bin/vllm", "Qwen/Qwen2.5-7B-Instruct-AWQ", served_name="teacher", port=8001,
                        enforce_eager=True)
    assert cmd[:3] == ["/env/bin/vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    assert cmd[cmd.index("--dtype") + 1] == "half"
    assert cmd[cmd.index("--served-model-name") + 1] == "teacher"
    assert "--enforce-eager" in cmd and cmd[cmd.index("--port") + 1] == "8001"
