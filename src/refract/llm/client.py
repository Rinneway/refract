"""One client for any OpenAI-compatible endpoint.

What it does:
- tries the base URLs in order and moves to the next one on a network error or HTTP 5xx;
- asks for structured output by a Pydantic schema (`json_schema`); if the provider rejects it, falls back to
  `json_object` and then to the schema written into the prompt — so small local models work too;
- validates the answer and asks the model once to repair invalid JSON;
- takes the cost from the provider (LiteLLM header `x-litellm-response-cost`, OpenRouter `usage.cost`) or, if
  there is none, from the price table in `refract.toml`;
- caches answers on disk by request hash — a repeated run costs nothing and is reproducible;
- writes every call (step, model, tokens, cost, latency, error) to a JSONL log.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ValidationError

from ..config import Settings, load_settings

T = TypeVar("T", bound=BaseModel)
STRUCTURED_MODES = ("json_schema", "json_object", "prompt")


@dataclass
class ChatResult:
    text: str
    model: str
    base_url: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class JsonResult(Generic[T]):
    """A schema-validated answer: `value` is the object or None; `calls` include the repair call."""

    value: T | None
    error: str | None
    calls: list[ChatResult]

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.calls if not c.cached)

    @property
    def tokens_in(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def latency_ms(self) -> int:
        return sum(c.latency_ms for c in self.calls)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    """What the agent needs from a model client. `LLMClient` and `ScriptedLLM` implement it."""

    settings: Settings

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        step: str | None = None,
        timeout_s: float | None = None,
    ) -> JsonResult[T]: ...


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        if not self.settings.api_key:
            raise LLMError(
                "No LLM_API_KEY: put it into .env (any non-empty value works for local Ollama or vLLM)"
            )
        self.cache_dir = Path(self.settings.cache_dir)
        self.log_path = Path(self.settings.log_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._clients = {
            url: OpenAI(
                api_key=self.settings.api_key, base_url=url, timeout=self.settings.timeout_s, max_retries=1
            )
            for url in self.settings.base_urls
        }
        self._mode: dict[str, str] = {}  # structured-output mode that works for a model id

    # ------------------------------------------------------------------ public API

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        use_cache: bool = True,
        step: str | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        """One model call. `model` is an alias from the settings (`main`, `verifier`) or a provider model id."""
        spec = self.settings.spec(model)
        payload: dict[str, Any] = {
            "model": spec.id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if spec.extra_body:
            payload["extra_body"] = spec.extra_body

        key = _cache_key(payload)
        if use_cache and (hit := self._cache_get(key)) is not None:
            hit.cached, hit.latency_ms = True, 0
            self._log(hit, step=step, error=None)
            return hit

        last_err: Exception | None = None
        for url, client in self._clients.items():
            t0 = time.perf_counter()
            try:
                api = client.with_options(timeout=timeout_s, max_retries=0) if timeout_s else client
                raw = api.chat.completions.with_raw_response.create(**payload)
                resp = raw.parse()
            except (APIConnectionError, APITimeoutError) as e:
                last_err = e
                continue
            except APIStatusError as e:
                last_err = e
                if e.status_code >= 500:
                    continue  # the endpoint is down — try the next one
                self._log_error(spec.id, url, step, e)
                raise LLMError(f"{spec.id}: HTTP {e.status_code}: {e.message}") from e
            latency = int((time.perf_counter() - t0) * 1000)
            usage = resp.usage
            dump = resp.model_dump()
            details = getattr(usage, "completion_tokens_details", None)
            result = ChatResult(
                text=(resp.choices[0].message.content or "").strip() if resp.choices else "",
                model=spec.id,
                base_url=url,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
                latency_ms=latency,
                raw=dump,
            )
            result.cost = self._cost(raw.headers.get("x-litellm-response-cost"), dump, result, spec)
            if use_cache and result.text:
                self._cache_put(key, result)
            self._log(result, step=step, error=None)
            return result

        self._log_error(spec.id, None, step, last_err)
        raise LLMError(f"{spec.id}: no endpoint answered: {last_err}") from last_err

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        repair: bool = True,
        use_cache: bool = True,
        step: str | None = None,
        timeout_s: float | None = None,
    ) -> JsonResult[T]:
        """An answer that must validate against a Pydantic schema; one repair attempt on failure."""
        model_id = self.settings.spec(model).id
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_cache": use_cache,
            "timeout_s": timeout_s,
        }
        first, mode = self._structured_call(model, model_id, messages, schema, step, kwargs)
        calls = [first]
        value, error = parse_json(schema, first.text)
        if value is None and repair:
            fix = [
                *_with_schema(messages, schema, mode),
                {"role": "assistant", "content": first.text[:8000]},
                {
                    "role": "user",
                    "content": f"The answer failed schema validation: {error}. Return the corrected JSON only.",
                },
            ]
            second = self.chat(
                model, fix, step=f"{step or 'json'}:repair", response_format=_format(schema, mode), **kwargs
            )
            calls.append(second)
            value, error = parse_json(schema, second.text)
        return JsonResult(value=value, error=error, calls=calls)

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embeddings from the first endpoint that answers (for hybrid retrieval)."""
        model = model or self.settings.embedding_model
        if not model:
            raise LLMError("No embedding model: set REFRACT_EMBEDDING_MODEL or [retrieval].embedding_model")
        last_err: Exception | None = None
        for client in self._clients.values():
            try:
                resp = client.embeddings.create(model=model, input=texts)
                return [d.embedding for d in resp.data]
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                last_err = e
        raise LLMError(f"embeddings failed: {last_err}") from last_err

    def list_models(self) -> dict[str, list[str] | str]:
        """Models available on every endpoint (a free request)."""
        out: dict[str, list[str] | str] = {}
        for url, client in self._clients.items():
            try:
                out[url] = sorted(m.id for m in client.models.list().data)
            except Exception as e:  # noqa: BLE001
                out[url] = f"error: {e}"
        return out

    # ------------------------------------------------------------------ structured output

    def _structured_call(self, model, model_id, messages, schema, step, kwargs) -> tuple[ChatResult, str]:
        start = STRUCTURED_MODES.index(self._mode.get(model_id, "json_schema"))
        last: LLMError | None = None
        for mode in STRUCTURED_MODES[start:]:
            try:
                res = self.chat(
                    model,
                    _with_schema(messages, schema, mode),
                    step=step,
                    response_format=_format(schema, mode),
                    **kwargs,
                )
                self._mode[model_id] = mode
                return res, mode
            except LLMError as e:
                last = e
                if not _format_rejected(e):
                    raise
        raise last or LLMError("structured output is not supported")

    # ------------------------------------------------------------------ cost, cache, log

    def _cost(self, header: str | None, dump: dict, r: ChatResult, spec) -> float:
        reported = _to_float(header)
        if reported:
            return reported
        usage_cost = (dump.get("usage") or {}).get("cost")
        if isinstance(usage_cost, (int, float)) and usage_cost > 0:
            return float(usage_cost)
        if spec.price_in is not None or spec.price_out is not None:
            return (
                r.prompt_tokens * (spec.price_in or 0) + r.completion_tokens * (spec.price_out or 0)
            ) / 1e6
        return 0.0

    def _cache_get(self, key: str) -> ChatResult | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return ChatResult(**json.loads(path.read_text(encoding="utf-8")))

    def _cache_put(self, key: str, result: ChatResult) -> None:
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8"
        )

    def _log(self, r: ChatResult, *, step: str | None, error: str | None) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": step,
            "model": r.model,
            "base_url": r.base_url,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "cost": 0.0 if r.cached else r.cost,
            "currency": self.settings.currency,
            "latency_ms": r.latency_ms,
            "cached": r.cached,
            "error": error,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _log_error(self, model_id: str, url: str | None, step: str | None, err: Exception | None) -> None:
        self._log(ChatResult(text="", model=model_id, base_url=url), step=step, error=repr(err))


# ---------------------------------------------------------------------- helpers


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _format(schema: type[BaseModel], mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False},
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def _with_schema(messages: list[dict[str, str]], schema: type[BaseModel], mode: str) -> list[dict[str, str]]:
    """Without native structured output the schema goes into the system prompt."""
    if mode == "json_schema":
        return messages
    note = (
        "\n\nReturn a single JSON object that validates against this JSON Schema, without any other text:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + note}, *messages[1:]]
    return [{"role": "system", "content": note.strip()}, *messages]


def _format_rejected(e: LLMError) -> bool:
    """HTTP 400/422 — most often the provider does not support this response_format; try a simpler mode."""
    text = str(e).lower()
    return "http 400" in text or "http 422" in text


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json(schema: type[T], text: str) -> tuple[T | None, str | None]:
    """Extract JSON from a model answer (strips ``` fences and stray text) and validate it."""
    cleaned = _FENCE.sub("", (text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None, "no JSON object in the answer"
    try:
        return schema.model_validate_json(cleaned[start : end + 1]), None
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:10])
        return None, problems or str(e)[:500]


def _to_float(value: str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except ValueError:
        return 0.0
