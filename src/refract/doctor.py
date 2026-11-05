"""`refract check` — is the configured endpoint good enough to run the agent?

For every model alias the agent uses (`main`, `verifier`, plus any you name) it checks, with three short requests:

1. the endpoint answers and knows the model id;
2. structured output works, and in which mode (`json_schema`, `json_object` or the schema in the prompt);
3. the context window holds an agent-sized prompt: a code word is placed at the very start of a ~9 000-token prompt
   and the model must return it. Ollama with a small `num_ctx` drops the start of the prompt and fails this test;
4. hidden reasoning: reasoning tokens spent on a trivial answer mean every step will be slow.

The answers are not cached, so the check always talks to the real endpoint.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from .llm.client import LLM, LLMClient, LLMError

FILLER = [
    "Совет директоров обсуждал денежно-кредитную политику и динамику цен на услуги.",
    "Exporters reported longer payment terms and higher freight costs in the third quarter.",
    "Розничные сети отмечают замедление спроса на товары длительного пользования.",
    "The central bank kept its guidance unchanged and pointed to sticky core inflation.",
    "Застройщики переносят старты новых проектов из-за дорогой ипотеки.",
    "Small businesses on the simplified tax regime are recalculating their prices for next year.",
]


class Echo(BaseModel):
    answer: str
    number: int


class Code(BaseModel):
    code: str


@dataclass
class ModelReport:
    alias: str
    model_id: str
    lines: list[tuple[str, str]] = field(default_factory=list)  # (status, text): OK | WARN | FAIL

    def add(self, status: str, text: str) -> None:
        self.lines.append((status, text))

    @property
    def ok(self) -> bool:
        return all(s != "FAIL" for s, _ in self.lines)


def long_prompt(code: str, target_tokens: int = 9000) -> list[dict[str, str]]:
    rng = random.Random(7)
    parts = [f"The code word is {code}. Remember it."]
    while sum(len(p) for p in parts) < target_tokens * 4:
        parts.append(rng.choice(FILLER))
    parts.append(
        "Question: what is the code word from the very first sentence? Return JSON with the field code."
    )
    return [{"role": "user", "content": " ".join(parts)}]


def check_model(client: LLMClient, alias: str, context_tokens: int = 9000) -> ModelReport:
    model_id = client.settings.model_id(alias)
    rep = ModelReport(alias, model_id)
    route = client.settings.route("draft_insight")

    t0 = time.perf_counter()
    try:
        res = client.chat_json(
            alias,
            [{"role": "user", "content": 'Return JSON: answer = "ok", number = 42.'}],
            Echo,
            max_tokens=400,
            use_cache=False,
            step="check:json",
            timeout_s=route.timeout_s,
        )
    except LLMError as e:
        rep.add("FAIL", f"no answer: {e}")
        if hint := error_hint(str(e), client):
            rep.add("HINT", hint)
        return rep
    seconds = time.perf_counter() - t0
    mode = client._mode.get(model_id, "json_schema")
    if res.value and res.value.number == 42:
        rep.add(
            "OK" if mode == "json_schema" else "WARN",
            f"structured output works (mode {mode}, {seconds:.1f} s)",
        )
    else:
        rep.add("FAIL", f"no schema-valid JSON: {res.error}")
    reasoning = sum(c.reasoning_tokens for c in res.calls)
    if reasoning > 50:
        rep.add(
            "WARN",
            f"{reasoning} hidden reasoning tokens on a trivial answer — switch reasoning off in extra_body",
        )

    code = f"cobalt-{random.randint(100, 999)}"
    t0 = time.perf_counter()
    try:
        res = client.chat_json(
            alias,
            long_prompt(code, context_tokens),
            Code,
            max_tokens=200,
            use_cache=False,
            step="check:context",
            timeout_s=max(route.timeout_s, 120),
        )
    except LLMError as e:
        rep.add("FAIL", f"long prompt failed: {e}")
        return rep
    seconds = time.perf_counter() - t0
    tokens = res.calls[0].prompt_tokens if res.calls else 0
    if res.value and code in res.value.code:
        rep.add("OK", f"context holds ~{context_tokens} tokens (reported {tokens}, {seconds:.1f} s)")
    else:
        hint = (
            " — the start of the prompt was dropped; Ollama: setx OLLAMA_CONTEXT_LENGTH 16384 and restart it"
        )
        rep.add("FAIL", f"lost the code word in a ~{context_tokens}-token prompt (reported {tokens}){hint}")
    for w in res.warnings:
        rep.add("WARN", w)
    return rep


def _is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))


def error_hint(error: str, client: LLM) -> str | None:
    """What to do about a failed call: a placeholder key on a remote endpoint, a missing model, no server."""
    s, url = client.settings, client.settings.base_urls[0]
    if "HTTP 402" in error:
        return f"{url} wants paid usage for this model (credits or a plan) — choose a model your plan covers, or a local one"
    if "HTTP 401" in error or "HTTP 403" in error:
        if not _is_local(url) and (s.api_key or "").lower() in ("ollama", "none", "x", "-"):
            return f"{url} needs a real key, but LLM_API_KEY is a placeholder for a local server"
        return f"{url} rejected LLM_API_KEY — check the key, or that LLM_BASE_URL points where you intend"
    if "HTTP 404" in error and _is_local(url):
        return f"the local server does not have this model: ollama pull {s.model_id('main')}"
    if "no endpoint answered" in error and _is_local(url):
        return "nothing answers on the local port — is Ollama running? (tray icon, or `ollama serve`)"
    return None


def run_check(client: LLMClient, aliases: list[str] | None = None) -> tuple[bool, str]:
    s = client.settings
    lines = [*s.notes, f"endpoints: {', '.join(s.base_urls)}", f"timeout scale: {s.timeout_scale}"]
    available: set[str] = set()
    for url, models in client.list_models().items():
        if isinstance(models, list):
            available |= set(models)
            lines.append(f"  {url}: {len(models)} models")
        else:
            lines.append(f"  {url}: {models}")
    ok = True
    for alias in aliases or ["main", "verifier"]:
        rep = check_model(client, alias)
        if available and rep.model_id not in available and f"{rep.model_id}:latest" not in available:
            hint = (
                f"ollama pull {rep.model_id}"
                if _is_local(s.base_urls[0])
                else "check the id with `refract models`"
            )
            rep.lines.insert(0, ("WARN", f"the endpoint does not list this model id ({hint})"))
        lines.append(f"\n[{alias}] {rep.model_id}")
        lines += [f"  {status:4} {text}" for status, text in rep.lines]
        ok &= rep.ok
    lines.append("\nready" if ok else "\nnot ready: fix the FAIL lines above")
    return ok, "\n".join(lines)
