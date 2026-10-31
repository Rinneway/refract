"""Settings: defaults → `refract.toml` → environment variables (later wins).

Any OpenAI-compatible endpoint works: OpenAI, OpenRouter, a LiteLLM proxy, vLLM, Ollama, LM Studio. Models are
referred to by alias (`main`, `verifier`, or your own); each pipeline step is routed to an alias with its own
token limit, timeout and fallback. See `refract.example.toml` and docs/configuration.md.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .i18n import Language, norm_lang

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ModelSpec:
    """A model behind an alias. Prices are per 1M tokens and used only if the provider reports no cost."""

    id: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    price_in: float | None = None
    price_out: float | None = None


@dataclass(frozen=True)
class Route:
    """Which model alias serves a pipeline step, and with which limits."""

    model: str = "main"
    fallback: str | None = "verifier"
    max_tokens: int = 2048
    timeout_s: float = 90
    temperature: float = 0.0


DEFAULT_ROUTES: dict[str, Route] = {
    "classify_news": Route("main", None, 900, 30),
    "plan": Route("main", None, 1200, 30),
    "draft_insight": Route("main", "verifier", 4000, 90),
    "verify": Route("verifier", "main", 2500, 60),
    "revise": Route("main", "verifier", 4000, 90),
    "headline": Route("main", "verifier", 300, 20),
    "chat_plan": Route("main", None, 800, 25),
    "chat": Route("main", "verifier", 2000, 60),
    "extract_theses": Route("main", None, 2500, 90),
    "judge": Route("verifier", None, 2500, 120),
}


@dataclass(frozen=True)
class Institution:
    """Whose research forms the house view and whose products the action policy may offer.

    `name_gen` is the Russian genitive form (“Демо-банка”), used in phrases like «позиция Демо-банка».
    """

    name: str = "Demo Bank"
    name_ru: str = "Демо-банк"
    name_gen: str = "Демо-банка"

    def display(self, lang: str) -> str:
        return self.name_ru if norm_lang(lang) == "ru" else self.name

    def genitive(self, lang: str) -> str:
        return self.name_gen if norm_lang(lang) == "ru" else self.name


@dataclass(frozen=True)
class Settings:
    language: Language = "ru"
    api_key: str | None = None
    base_urls: tuple[str, ...] = (DEFAULT_BASE_URL,)
    timeout_s: float = 120
    cache_dir: Path = Path(".cache/llm")
    log_path: Path = Path("logs/llm_calls.jsonl")
    currency: str = "USD"
    models: dict[str, ModelSpec] = field(default_factory=dict)
    routes: dict[str, Route] = field(default_factory=lambda: dict(DEFAULT_ROUTES))
    embedding_model: str | None = None
    institution: Institution = field(default_factory=Institution)
    offer_products: bool = True
    data_dir: Path = Path("examples/demo")
    roles_file: Path | None = None  # your own roles.json instead of the packaged one
    actions_file: Path | None = None  # your own actions.json (action policy)

    def model_id(self, alias: str) -> str:
        """Alias → provider model id. An unknown alias is treated as a model id itself."""
        spec = self.models.get(alias)
        return spec.id if spec else alias

    def spec(self, alias_or_id: str) -> ModelSpec:
        if alias_or_id in self.models:
            return self.models[alias_or_id]
        for spec in self.models.values():
            if spec.id == alias_or_id:
                return spec
        return ModelSpec(id=alias_or_id)

    def route(self, step: str) -> Route:
        base = step.split(":", 1)[0]
        return self.routes.get(base, Route())

    def with_language(self, lang: str | None) -> Settings:
        return replace(self, language=norm_lang(lang or self.language))


def _models_from(raw: dict[str, Any]) -> dict[str, ModelSpec]:
    return {
        alias: ModelSpec(
            id=str(m["id"]),
            extra_body=dict(m.get("extra_body") or {}),
            price_in=m.get("price_in"),
            price_out=m.get("price_out"),
        )
        for alias, m in raw.items()
    }


def _routes_from(raw: dict[str, Any]) -> dict[str, Route]:
    routes = dict(DEFAULT_ROUTES)
    for step, r in raw.items():
        base = routes.get(step, Route())
        routes[step] = replace(
            base,
            **{k: r[k] for k in ("model", "fallback", "max_tokens", "timeout_s", "temperature") if k in r},
        )
        if r.get("fallback") in ("", "none", False):
            routes[step] = replace(routes[step], fallback=None)
    return routes


def load_settings(config_path: str | Path | None = None, *, env_file: str | Path | None = None) -> Settings:
    """Read `refract.toml` (or `REFRACT_CONFIG`) and environment variables (`.env` is loaded first)."""
    load_dotenv(env_file) if env_file else load_dotenv()
    path = Path(config_path or os.getenv("REFRACT_CONFIG", "refract.toml"))
    raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    llm = raw.get("llm", {})
    inst = raw.get("institution", {})
    actions = raw.get("actions", {})
    policies = raw.get("policies", {})
    models = _models_from(raw.get("models", {}))

    main = models.get("main", ModelSpec(DEFAULT_MODEL))
    if env_model := os.getenv("REFRACT_MODEL"):
        main = replace(main, id=env_model)
    verifier = models.get("verifier", main)  # no verifier configured — the same model checks itself
    if env_verifier := os.getenv("REFRACT_VERIFIER_MODEL"):
        verifier = replace(verifier, id=env_verifier)
    models["main"], models["verifier"] = main, verifier

    urls = os.getenv("LLM_BASE_URL") or os.getenv("LLM_BASE_URLS")
    base_urls = (
        tuple(u.strip() for u in urls.split(",") if u.strip())
        if urls
        else tuple(llm.get("base_urls") or [DEFAULT_BASE_URL])
    )
    return Settings(
        language=norm_lang(os.getenv("REFRACT_LANGUAGE") or raw.get("language") or "ru"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or llm.get("api_key"),
        base_urls=base_urls,
        timeout_s=float(os.getenv("LLM_TIMEOUT_S") or llm.get("timeout_s", 120)),
        cache_dir=Path(os.getenv("LLM_CACHE_DIR") or llm.get("cache_dir", ".cache/llm")),
        log_path=Path(os.getenv("LLM_LOG_PATH") or llm.get("log_path", "logs/llm_calls.jsonl")),
        currency=str(llm.get("currency", "USD")),
        models=models,
        routes=_routes_from(raw.get("routes", {})),
        embedding_model=os.getenv("REFRACT_EMBEDDING_MODEL")
        or raw.get("retrieval", {}).get("embedding_model"),
        institution=Institution(
            name=inst.get("name", Institution.name),
            name_ru=inst.get("name_ru", inst.get("name", Institution.name_ru)),
            name_gen=inst.get("name_gen", inst.get("name_ru", inst.get("name", Institution.name_gen))),
        ),
        offer_products=bool(actions.get("offer_products", True)),
        data_dir=Path(os.getenv("REFRACT_DATA_DIR") or raw.get("data_dir", "examples/demo")),
        roles_file=Path(policies["roles"]) if policies.get("roles") else None,
        actions_file=Path(policies["actions"]) if policies.get("actions") else None,
    )
