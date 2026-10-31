"""A scripted model for tests, demos without a key and reproducible examples.

`ScriptedLLM` returns prepared objects by pipeline step name (`plan`, `draft_insight`, `verify`…). The last answer
for a step repeats, so a single verdict can serve every verification round. The key `"*"` answers any other step;
an answer may be a callable `(messages, schema) -> object` that builds the reply from the request.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..config import Settings
from .client import ChatResult, JsonResult

Answer = BaseModel | dict | None | Callable[[list[dict[str, str]], type[BaseModel]], Any]


class ScriptedLLM:
    def __init__(self, responses: dict[str, list[Answer]], settings: Settings | None = None) -> None:
        self.settings = settings or Settings(api_key="scripted")
        self.responses = {k: list(v) for k, v in responses.items()}
        self.steps: list[str] = []
        self.messages: dict[str, list[dict[str, str]]] = {}

    def chat_json(self, model, messages, schema, *, step=None, **_) -> JsonResult:
        step = step or "json"
        self.steps.append(step)
        self.messages[step] = messages
        queue = self.responses.get(step.split(":")[0]) or self.responses.get("*", [None])
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(answer) and not isinstance(answer, BaseModel):
            answer = answer(messages, schema)
        if isinstance(answer, dict):
            answer = schema.model_validate(answer)
        call = ChatResult(
            text="{}",
            model=self.settings.spec(model).id,
            base_url="scripted",
            prompt_tokens=100,
            completion_tokens=40,
            cost=0.001,
            latency_ms=5,
        )
        return JsonResult(
            value=answer, error=None if answer is not None else "no scripted answer", calls=[call]
        )
