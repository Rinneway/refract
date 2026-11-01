"""A real HTTP server that speaks the OpenAI chat API, for end-to-end tests of the CLI and the LLM client.

It answers by the name of the requested JSON schema (`response_format.json_schema.name`), with a fixed value or a
function of the messages, and records every request. Unknown schemas get HTTP 500, so a missing answer is loud.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Answer = dict[str, Any] | Callable[[list[dict[str, str]]], dict[str, Any]]


class FakeOpenAI:
    def __init__(
        self, answers: dict[str, Answer], models: tuple[str, ...] = ("fake-main", "fake-verifier")
    ) -> None:
        self.answers, self.models = answers, models
        self.requests: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:  # keep test output clean
                pass

            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.rstrip("/").endswith("/models"):
                    self._send(
                        200, {"object": "list", "data": [{"id": m, "object": "model"} for m in server.models]}
                    )
                else:
                    self._send(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                server.requests.append(body)
                name = ((body.get("response_format") or {}).get("json_schema") or {}).get("name", "")
                answer = server.answers.get(name)
                if answer is None:
                    self._send(500, {"error": {"message": f"fake server has no answer for schema {name!r}"}})
                    return
                value = answer(body["messages"]) if callable(answer) else answer
                text = json.dumps(value, ensure_ascii=False)
                prompt = sum(len(m.get("content") or "") for m in body["messages"]) // 4
                self._send(
                    200,
                    {
                        "id": f"fake-{len(server.requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt,
                            "completion_tokens": len(text) // 4,
                            "total_tokens": prompt + len(text) // 4,
                        },
                    },
                )

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def schemas(self) -> list[str]:
        return [
            ((r.get("response_format") or {}).get("json_schema") or {}).get("name", "") for r in self.requests
        ]

    def __enter__(self) -> FakeOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
