"""LLMClient against a fake OpenAI endpoint: structured-output fallback, cost, cache, endpoint failover."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from refract.config import ModelSpec, Settings
from refract.llm import LLMClient, LLMError, parse_json


class Answer(BaseModel):
    value: int


class FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    @property
    def with_raw_response(self):
        return self

    def create(self, **payload):
        self.calls.append(payload)
        return self.behaviour(payload)


def raw(text: str, headers=None, usage_cost=None):
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, completion_tokens_details=None)
    dump = {"usage": {"cost": usage_cost} if usage_cost else {}}
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage, model_dump=lambda: dump
    )
    return SimpleNamespace(headers=headers or {}, parse=lambda: resp)


def bad_request(msg="response_format json_schema is not supported"):
    req = httpx.Request("POST", "http://fake/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def client(tmp_path, behaviour, **settings_kw):
    s = Settings(
        api_key="k",
        base_urls=("http://a/v1", "http://b/v1"),
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "log.jsonl",
        models={"main": ModelSpec("m-1", price_in=1.0, price_out=2.0)},
        **settings_kw,
    )
    c = LLMClient(s)
    fakes = {}
    for url in c._clients:
        fake = FakeCompletions(behaviour(url))
        fakes[url] = fake
        c._clients[url] = SimpleNamespace(
            chat=SimpleNamespace(completions=fake), with_options=lambda **_: None
        )
    for url, fake in fakes.items():
        c._clients[url].with_options = (
            lambda f: lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=f))
        )(fake)
    return c, fakes


def test_structured_output_falls_back_to_json_object(tmp_path) -> None:
    def behaviour(url):
        def create(payload):
            if payload.get("response_format", {}).get("type") == "json_schema":
                raise bad_request()
            return raw('```json\n{"value": 7}\n```')

        return create

    c, fakes = client(tmp_path, behaviour)
    res = c.chat_json(
        "main", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], Answer, step="t"
    )
    assert res.value == Answer(value=7)
    sent = fakes["http://a/v1"].calls[-1]
    assert (
        sent["response_format"] == {"type": "json_object"} and "JSON Schema" in sent["messages"][0]["content"]
    )
    assert res.cost == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)  # price table, no provider cost


def test_provider_cost_cache_and_log(tmp_path) -> None:
    c, fakes = client(
        tmp_path, lambda url: lambda p: raw('{"value": 1}', headers={"x-litellm-response-cost": "0.05"})
    )
    msgs = [{"role": "user", "content": "u"}]
    first = c.chat_json("main", msgs, Answer, step="t")
    second = c.chat_json("main", msgs, Answer, step="t")
    assert first.cost == pytest.approx(0.05) and second.cost == 0 and second.calls[0].cached
    assert len(fakes["http://a/v1"].calls) == 1
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[0]["cost"] == pytest.approx(0.05) and log[1]["cached"] is True


def test_failover_to_next_endpoint(tmp_path) -> None:
    def behaviour(url):
        def create(payload):
            if url == "http://a/v1":
                raise openai.APIConnectionError(request=httpx.Request("POST", url))
            return raw('{"value": 2}', usage_cost=0.01)

        return create

    c, _ = client(tmp_path, behaviour)
    res = c.chat_json("main", [{"role": "user", "content": "u"}], Answer, use_cache=False)
    assert res.value.value == 2 and res.calls[0].base_url == "http://b/v1" and res.cost == pytest.approx(0.01)


def test_repair_round(tmp_path) -> None:
    answers = iter(['{"value": "x"}', '{"value": 3}'])
    c, _ = client(tmp_path, lambda url: lambda p: raw(next(answers)))
    res = c.chat_json("main", [{"role": "user", "content": "u"}], Answer, use_cache=False)
    assert res.value.value == 3 and len(res.calls) == 2


def test_no_key() -> None:
    with pytest.raises(LLMError):
        LLMClient(Settings(api_key=None))


def test_parse_json() -> None:
    assert parse_json(Answer, 'Here: {"value": 5} done')[0] == Answer(value=5)
    assert parse_json(Answer, "nothing")[0] is None


def raw_choice(text: str, prompt_tokens: int = 1000, finish_reason: str = "stop"):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=50, completion_tokens_details=None)
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], usage=usage, model_dump=lambda: {"usage": {}})
    return SimpleNamespace(headers={}, parse=lambda: resp)


def test_truncated_context_is_reported(tmp_path) -> None:
    from refract.llm.client import context_warning

    long = [{"role": "user", "content": "слово " * 8000}]  # ~48k characters ≈ 10k+ tokens
    assert context_warning(long, 4096) and context_warning(long, 12000) is None
    assert context_warning([{"role": "user", "content": "short"}], 10) is None
    c, _ = client(tmp_path, lambda url: lambda p: raw_choice('{"value": 1}', prompt_tokens=4096))
    res = c.chat_json("main", long, Answer, use_cache=False)
    assert res.value.value == 1 and any("OLLAMA_CONTEXT_LENGTH" in w for w in res.warnings)


def test_cut_answer_is_retried_with_more_tokens(tmp_path) -> None:
    seen = []

    def behaviour(url):
        def create(payload):
            seen.append(payload["max_tokens"])
            if len(seen) == 1:
                return raw_choice('{"val', finish_reason="length")
            return raw_choice('{"value": 9}')

        return create

    c, _ = client(tmp_path, behaviour)
    res = c.chat_json("main", [{"role": "user", "content": "u"}], Answer, max_tokens=500, use_cache=False)
    assert res.value.value == 9 and seen == [500, 1000]
    assert any("max_tokens=500" in w for w in res.warnings)


def test_timeout_scale() -> None:
    s = Settings(timeout_scale=3)
    assert s.route("plan").timeout_s == 90 and s.route("verify:fallback").timeout_s == 180


def test_check_reports_lost_context(tmp_path) -> None:
    import re

    from refract.doctor import run_check

    def behaviour(url):
        def create(payload):
            text = payload["messages"][-1]["content"]
            if "code word" in text:
                if len(text) > 20000:  # a server with a small context window "forgets" the start
                    return raw_choice('{"code": "unknown"}', prompt_tokens=4096)
                code = re.search(r"cobalt-\d+", text).group(0)
                return raw_choice(json.dumps({"code": code}))
            return raw_choice('{"answer": "ok", "number": 42}')

        return create

    c, _ = client(tmp_path, behaviour)
    c.list_models = lambda: {"http://a/v1": ["m-1"]}
    ok, report = run_check(c, ["main"])
    assert not ok and "lost the code word" in report and "OLLAMA_CONTEXT_LENGTH" in report
    assert "structured output works" in report


def test_check_explains_a_placeholder_key_on_a_remote_endpoint(tmp_path, monkeypatch) -> None:
    from refract.config import load_settings
    from refract.doctor import run_check

    toml = tmp_path / "refract.toml"
    toml.write_text('[llm]\nbase_urls = ["http://localhost:11434/v1"]\n', encoding="utf-8")
    monkeypatch.setenv("LLM_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    s = load_settings(toml, env_file=tmp_path / ".env")
    assert any("overrides [llm].base_urls" in n for n in s.notes)

    def unauthorized(payload):
        req = httpx.Request("POST", "https://ollama.com/v1/chat/completions")
        raise openai.AuthenticationError("Unauthorized", response=httpx.Response(401, request=req), body=None)

    c = LLMClient(s)
    for url in c._clients:
        fake = FakeCompletions(unauthorized)
        c._clients[url] = SimpleNamespace(
            chat=SimpleNamespace(completions=fake),
            with_options=lambda f=fake, **_: SimpleNamespace(chat=SimpleNamespace(completions=f)),
        )
    c.list_models = lambda: {"https://ollama.com/v1": ["glm-5.3-flash"]}
    ok, report = run_check(c, ["main"])
    assert not ok and "placeholder" in report and "overrides" in report
