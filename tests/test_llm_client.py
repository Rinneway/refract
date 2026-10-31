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
