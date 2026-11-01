import json
import re

from refract.cli import main
from refract.schemas import Profile
from tests.conftest import DEMO, ROOT


def test_cli_reports_bad_input_without_traceback(capsys, tmp_path) -> None:
    base = ["analyze", "--data", str(DEMO), "--profile", "demo_coffee_ip"]
    assert main([*base[:4], "nobody", "--event", "ev_rate_hold_ru"]) == 2
    assert main([*base, "--event", "nope"]) == 2
    assert main([*base, "--event", "ev_rate_hold_ru", "--ablation", "bogus"]) == 2
    assert main([*base, "--title", "t", "--text-file", str(tmp_path / "missing.txt")]) == 2
    assert main(["chat", "--run", str(tmp_path / "missing.json"), "--q", "?"]) == 2
    err = capsys.readouterr().err
    assert "unknown profile" in err and "unknown event" in err and "unknown ablation" in err
    assert "Traceback" not in err


def test_cli_demo_and_lists(capsys) -> None:
    assert main(["demo", "rate"]) == 0
    out = capsys.readouterr().out
    assert "Как получен вывод" in out and "finalize" in out
    assert main(["profiles", "--data", str(DEMO)]) == 0
    assert "demo_london_studio" in capsys.readouterr().out


def test_schema_export(tmp_path) -> None:
    assert main(["schemas", "--out", str(tmp_path)]) == 0
    schema = json.loads((tmp_path / "insight.schema.json").read_text(encoding="utf-8"))
    assert "house_view" in schema["properties"]
    # docs/schemas is what integrators read: it must match the code (regenerate with `refract schemas`)
    stale = [
        p.name
        for p in sorted(tmp_path.glob("*.json"))
        if p.read_text(encoding="utf-8") != (ROOT / "docs" / "schemas" / p.name).read_text(encoding="utf-8")
    ]
    assert not stale, f"run `refract schemas`: {stale}"


def test_demo_profiles_validate() -> None:
    kinds = {
        Profile.model_validate_json(p.read_text(encoding="utf-8")).kind.value
        for p in (DEMO / "profiles").glob("*.json")
    }
    assert kinds == {"company", "sole_proprietor", "self_employed", "individual"}


def test_no_confidential_material_in_repository() -> None:
    """Guard for the public repository: no client ids of the original dataset format, no keys, no .env."""
    bad = [
        re.compile(r"\bU_\d{2,}\b"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9]{16,}"),
    ]
    skip = {".git", ".venv", ".cache", "__pycache__", ".ruff_cache", ".pytest_cache", "runs", "logs"}
    for path in ROOT.rglob("*"):
        if (
            path.is_dir()
            or any(part in skip for part in path.parts)
            or path.suffix in {".lock", ".png", ".pyc"}
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for rx in bad:
            assert not rx.search(text), f"{path.relative_to(ROOT)}: {rx.pattern}"
    assert not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text()
