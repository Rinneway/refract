"""Hand-written data folders: byte-order marks, Windows encodings, broken files, repeated ids."""

from __future__ import annotations

import json
import shutil

import pytest

from refract.cli import main
from refract.config import load_settings
from refract.eval import load_pairs
from refract.knowledge.store import DataStore
from refract.textutil import DataError, read_json
from tests.conftest import DEMO

BOM = "﻿"


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(DEMO, root)
    return root


def bom(path) -> None:  # what Windows Notepad does on “Save as UTF-8”
    path.write_text(BOM + path.read_text(encoding="utf-8"), encoding="utf-8")


def test_files_saved_with_a_byte_order_mark_load(data, tmp_path) -> None:
    for path in (
        data / "profiles" / "demo_coffee_ip.json",
        data / "pairs.jsonl",
        *(data / "knowledge" / "docs").iterdir(),
    ):
        bom(path)
    store = DataStore.from_dir(data)
    assert "demo_coffee_ip" in store.profiles and not store.problems
    assert {tuple(c.countries) for c in store.chunks} >= {("RU",)}  # front matter still read after the mark
    assert len(load_pairs(data / "pairs.jsonl")) > 0
    config = tmp_path / "refract.toml"
    config.write_text(BOM + '[llm]\nbase_urls = ["http://localhost:11434/v1"]\n', encoding="utf-8")
    assert load_settings(config).base_urls == ("http://localhost:11434/v1",)


def test_a_russian_text_saved_in_windows_1251_is_read(data) -> None:
    doc = data / "knowledge" / "docs" / "zz_cp1251.txt"
    doc.write_bytes(
        ("# Обзор\n\n" + "Банк ожидает снижения ставки до 12% к концу года, рост кредитования. " * 3).encode(
            "cp1251"
        )
    )
    store = DataStore.from_dir(data)
    assert any("снижения ставки" in c.text for c in store.chunks) and not store.problems


def test_a_broken_file_is_skipped_with_a_note_naming_it(data, capsys) -> None:
    (data / "profiles" / "broken.json").write_text('{"profile_id": "x",', encoding="utf-8")
    bad = json.loads((data / "profiles" / "demo_coffee_ip.json").read_text(encoding="utf-8"))
    (data / "profiles" / "typo.json").write_text(
        json.dumps({**bad, "profile_id": "typo", "kind": "companny"}), encoding="utf-8"
    )
    (data / "profiles" / "copy.json").write_text(
        json.dumps(bad), encoding="utf-8"
    )  # the same profile_id again
    (data / "knowledge" / "docs" / "scan.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    store = DataStore.from_dir(data)
    assert "demo_coffee_ip" in store.profiles and "typo" not in store.profiles
    notes = "\n".join(store.problems)
    assert "broken.json: not valid JSON" in notes and "line 1" in notes
    assert "typo.json [typo]: skipped — kind:" in notes
    assert "profile demo_coffee_ip appears 2 times" in notes and "scan.pdf" in notes

    assert main(["profiles", "--data", str(data)]) == 0  # the CLI still works and says what it skipped
    err = capsys.readouterr().err
    assert "broken.json" in err and "typo.json" in err


def test_broken_config_is_a_message_not_a_traceback(tmp_path, capsys) -> None:
    config = tmp_path / "refract.toml"
    config.write_text("[llm\nbase_urls = 1\n", encoding="utf-8")
    assert main(["--config", str(config), "profiles"]) == 2
    assert f"refract: {config}: not valid TOML" in capsys.readouterr().err
    with pytest.raises(DataError, match="not valid JSON"):
        read_json(config)
