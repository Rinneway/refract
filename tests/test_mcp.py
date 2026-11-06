import asyncio

import pytest

pytest.importorskip("mcp")

from refract.demo import OK, RATE_PLAN, _rate_draft
from refract.facade import Refract
from refract.llm import ScriptedLLM
from refract.mcp_server import build_server


def text_of(result) -> str:
    result = result[0] if isinstance(result, tuple) else result
    content = getattr(result, "content", result)  # mcp 2: CallToolResult; mcp 1: a list of content blocks
    return "\n".join(getattr(c, "text", "") for c in content)


def test_mcp_tools(store, settings) -> None:
    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    server = build_server(Refract(store, llm=llm, settings=settings))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"list_profiles", "analyze_news", "ask_followup", "get_trace", "search_research"} <= names
    assert "demo_coffee_ip" in text_of(asyncio.run(server.call_tool("list_profiles", {})))
    out = text_of(
        asyncio.run(
            server.call_tool(
                "analyze_news", {"profile_id": "demo_building_materials", "event_id": "ev_rate_hold_ru"}
            )
        )
    )
    assert "insight_id" in out and "Позиция Демо-банка" in out


def test_mcp_in_demo_mode_says_what_needs_a_model() -> None:
    server = build_server(Refract.demo())
    ok = text_of(
        asyncio.run(
            server.call_tool("analyze_news", {"profile_id": "demo_grain_farm", "event_id": "ev_wheat_en"})
        )
    )
    assert "insight_id" in ok
    try:
        res = asyncio.run(
            server.call_tool("analyze_news", {"profile_id": "demo_grain_farm", "event_id": "ev_rate_hold_ru"})
        )
        msg = text_of(res)
    except Exception as e:  # noqa: BLE001 — mcp 1.x raises a ToolError
        msg = str(e)
    assert "replayable pairs" in msg
