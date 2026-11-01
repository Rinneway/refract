"""Reader roles and the action policy, read from JSON (`refract/policies/` or your own files).

A new role or rule is a JSON edit, not a code change. Point `[policies]` in `refract.toml` at your own files to
replace the defaults.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from ..i18n import norm_lang

POLICIES = Path(__file__).resolve().parents[1] / "policies"


def _loc(value: Any, lang: str) -> str:
    """{"ru": …, "en": …} → the string for the language; a plain string stays as is."""
    if isinstance(value, dict):
        return str(value.get(norm_lang(lang)) or next(iter(value.values()), ""))
    return "" if value is None else str(value)


@dataclass(frozen=True)
class RoleProfile:
    key: str
    raw: dict[str, Any] = field(repr=False, hash=False, compare=False)

    def name(self, lang: str) -> str:
        return _loc(self.raw["name"], lang)

    def main_question(self, lang: str) -> str:
        return _loc(self.raw["main_question"], lang)

    def depth(self, lang: str) -> str:
        return _loc(self.raw["depth"], lang)

    def tone(self, lang: str) -> str:
        return _loc(self.raw["tone"], lang)

    @property
    def focus(self) -> tuple[str, ...]:
        return tuple(self.raw.get("focus", []))

    @property
    def max_claims(self) -> int:
        return int(self.raw.get("max_claims_per_block", 3))


@dataclass(frozen=True)
class ActionRule:
    id: str
    kind: str
    when: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, hash=False, compare=False)
    product_key: str | None = None
    product_names: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

    def text(self, lang: str) -> str:
        return _loc(self.raw["text"], lang)

    def rationale(self, lang: str) -> str:
        return _loc(self.raw["rationale"], lang)

    def product(self, lang: str) -> str | None:
        return self.product_names.get(norm_lang(lang)) if self.product_key else None

    def note(self, lang: str) -> str | None:
        return _loc(self.when["note"], lang) if self.when.get("note") else None


@cache
def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Policy:
    """Roles and action rules. `Policy()` reads the defaults shipped with the package."""

    def __init__(self, roles_file: str | Path | None = None, actions_file: str | Path | None = None) -> None:
        self._roles = _read(str(roles_file or POLICIES / "roles.json"))
        self._actions = _read(str(actions_file or POLICIES / "actions.json"))

    def roles(self) -> dict[str, RoleProfile]:
        return {key: RoleProfile(key, r) for key, r in self._roles["roles"].items()}

    def role(self, key: str | None) -> RoleProfile:
        """Unknown role → the default role."""
        roles = self.roles()
        return roles.get(key or "", roles[self._roles["default_role"]])

    def rules(self) -> tuple[ActionRule, ...]:
        products = self._actions.get("products", {})
        out = []
        for r in self._actions["rules"]:
            key = r.get("product")
            names = products.get(key, {}) if key else {}
            out.append(
                ActionRule(
                    id=r["id"],
                    kind=r["kind"],
                    when=r.get("when", {}),
                    raw=r,
                    product_key=key,
                    product_names=names,
                )
            )
        return tuple(out)

    def product_patterns(self) -> dict[str, re.Pattern[str]]:
        """Product key → a pattern that names the product in free text (`product_patterns` in the JSON)."""
        raw = self._actions.get("product_patterns", {})
        return {k: re.compile(v, re.IGNORECASE) for k, v in raw.items() if not k.startswith("_")}

    def fallback(self, lang: str) -> dict[str, str]:
        fb = self._actions["fallback"]
        return {"kind": fb["kind"], "text": _loc(fb["text"], lang), "rationale": _loc(fb["rationale"], lang)}


DEFAULT_POLICY = Policy()
