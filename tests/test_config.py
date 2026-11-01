from refract.config import load_settings
from tests.conftest import ROOT


def test_example_toml_and_env_override(monkeypatch, tmp_path) -> None:
    for var in (
        "LLM_BASE_URL",
        "LLM_BASE_URLS",
        "REFRACT_MODEL",
        "REFRACT_VERIFIER_MODEL",
        "REFRACT_LANGUAGE",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("")
    s = load_settings(ROOT / "refract.example.toml", env_file=env)
    assert s.model_id("main") == "z-ai/glm-5.3-flash" and s.route("verify").model == "verifier"
    assert s.institution.genitive("ru") == "Демо-банка" and s.institution.display("en") == "Demo Bank"
    monkeypatch.setenv("REFRACT_MODEL", "local/llama")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1, http://backup/v1")
    monkeypatch.setenv("REFRACT_LANGUAGE", "English")
    s = load_settings(ROOT / "refract.example.toml", env_file=env)
    assert s.model_id("main") == "local/llama" and s.model_id("verifier") == "deepseek/deepseek-v4.1-flash"
    assert s.base_urls == ("http://localhost:11434/v1", "http://backup/v1") and s.language == "en"


def test_defaults_without_config(monkeypatch, tmp_path) -> None:
    for var in ("REFRACT_MODEL", "REFRACT_VERIFIER_MODEL", "LLM_BASE_URL", "LLM_BASE_URLS"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings(tmp_path / "missing.toml", env_file=tmp_path / ".env")
    assert s.model_id("main") == s.model_id("verifier") == "gpt-4o-mini"
    assert s.route("draft_insight:fallback").max_tokens == 4000


def test_cloud_config_reads_its_own_key(monkeypatch, tmp_path) -> None:
    for var in ("LLM_BASE_URL", "LLM_BASE_URLS", "REFRACT_MODEL", "REFRACT_VERIFIER_MODEL", "OLLAMA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "ollama")  # the local placeholder stays in .env
    env = tmp_path / ".env"
    env.write_text("")
    cloud = ROOT / "examples" / "config" / "ollama-cloud.toml"
    s = load_settings(cloud, env_file=env)
    assert s.api_key == "ollama" and any("OLLAMA_API_KEY is not set" in n for n in s.notes)
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-cloud")
    monkeypatch.setenv("REFRACT_VERIFIER_MODEL", "gemma4:12b")
    s = load_settings(cloud, env_file=env)
    assert s.api_key == "sk-cloud" and s.base_urls == ("https://ollama.com/v1",)
    assert any("REFRACT_VERIFIER_MODEL overrides" in n for n in s.notes)  # shown before `eval judge`
