"""`refract shell`: the interactive console, driven by a script."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from refract.cli import main
from refract.facade import Refract
from refract.shell import Shell, Style

SCRIPT = Path(__file__).parent / "fixtures" / "shell_script.txt"


@pytest.fixture(scope="module")
def rf() -> Refract:
    return Refract.demo()


def session(rf: Refract, commands: str, *, color: bool = False) -> tuple[Shell, str]:
    out = io.StringIO()
    sh = Shell(rf, stdin=io.StringIO(commands), stdout=out, color=color, interactive=False)
    sh.cmdloop()
    return sh, out.getvalue()


def test_the_ci_script_runs_without_a_failure(capsys) -> None:
    assert main(["shell", "--demo", "--script", str(SCRIPT)]) == 0
    out = capsys.readouterr().out
    assert "> analyze ev_sme_loans_ru" in out and "## " in out and "recorded:qwen3.5:9b" in out
    assert "Brief for demo_london_studio" in out


def test_numbers_refer_to_the_last_list(rf) -> None:
    sh, out = session(rf, "profiles\nuse 2\nfeed 10\n")
    assert sh.profile_id == list(rf.store.profiles)[1] == "demo_coffee_ip" and "▶ demo" in out
    n = sh.last.index("ev_sme_loans_ru") + 1
    sh, out = session(rf, f"use demo_coffee_ip\nfeed 10\nanalyze {n}\n")
    assert sh.run is not None and sh.run.insight is not None and not sh.failed
    _, out = session(rf, "use demo_coffee_ip\nnews\n")
    assert "ev_sme_loans_ru ▶ demo" in out and "ev_yuan_ru ▶" not in out
    sh, out = session(rf, "use demo_coffee_ip\nanalyze 1\n")  # no list shown yet
    assert sh.failed and ("номер" in out or "number" in out)


def test_mistakes_are_messages_not_crashes(rf) -> None:
    sh, out = session(
        rf,
        "analyze 1\nuse nobody\nuse demo_family_saver\nanalyze ev_nothing\nrole boss\nlang de\nfoo\n"
        "analyze ev_dollar_en\nask Что делать?\nsources\nload missing.json\n",
    )
    assert sh.failed
    for text in (
        "profiles",
        "nobody",
        "ev_nothing",
        "cfo",
        "ru, en",
        "foo",
        "replayable pairs",
        "missing.json",
    ):
        assert text in out, text
    assert "Traceback" not in out


def test_plain_text_after_an_insight_is_a_question(rf) -> None:
    asked: list[str] = []
    sh, _ = session(rf, "use demo_grain_farm\nanalyze ev_wheat_en\n")
    sh.do_ask = asked.append  # type: ignore[method-assign]
    sh.onecmd("what about the freight costs?")
    sh.onecmd("feeed")
    assert asked == ["what about the freight costs?"]


def test_paste_needs_a_model_in_demo_mode(rf) -> None:
    _, out = session(
        rf, "use demo_coffee_ip\npaste\nКофе дорожает\nЦены на арабику выросли на 20% за месяц.\n\n"
    )
    assert "pasted news text needs a model" in out


def test_save_and_load_a_run(rf, tmp_path) -> None:
    sh, _ = session(rf, f"use demo_grain_farm\nanalyze ev_wheat_en\nsave {tmp_path}\n")
    saved = next(tmp_path.glob("*.json"))
    sh2, _ = session(rf, f"load {saved}\nsources\n")
    assert sh2.profile_id == "demo_grain_farm" and sh2.run.insight.insight_id == sh.run.insight.insight_id


def test_colours_only_when_asked(rf) -> None:
    assert "\033[" not in session(rf, "profiles\n")[1]
    assert "\033[1m" in session(rf, "help\n", color=True)[1]
    assert Style(True).markdown("## Title\n**bold** and `code`").count("\033[0m") == 3
    assert Style(False).markdown("## Title") == "## Title"


def test_completion_lists_commands(rf) -> None:
    sh = Shell(rf, stdin=io.StringIO(""), stdout=io.StringIO(), interactive=False)
    assert sh.completenames("an") == ["analyze"]
    assert "EOF" not in sh.completenames("")
