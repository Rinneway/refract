# Contributing

Thanks for your interest in Refract. Bug reports, questions and pull requests are welcome.

## Setup

```bash
git clone https://github.com/Rinneway/refract.git
cd refract
uv sync --all-extras
uv run pytest            # no network or API key needed: tests use a scripted model
uv run ruff check src tests && uv run ruff format src tests
```

## Ground rules

- **No real or confidential data.** Profiles, news, research and examples must be synthetic or clearly licensed. Never commit `.env`, keys, client data or texts you have no right to publish. `tests/test_cli_and_schemas.py` guards against the obvious cases.
- **Code bounds the model.** If a rule can be checked deterministically (a source exists, a number exists, a product is allowed), check it in code, not in a prompt.
- **Both languages.** A change to a prompt, a label or a policy text needs the Russian and the English version; `tests/test_prompts.py` checks that English prompts contain no Russian text.
- **Version prompts.** Bump `VERSION` in a prompt module when its text changes; versions are written into traces and evaluation results.
- **Tests with the scripted model.** New behaviour of the agent gets a test with `refract.llm.ScriptedLLM`; see `tests/test_agent.py` and `refract/demo.py`.
- **Measure quality changes.** For changes that affect what the reader sees, run `refract eval` on a set of pairs before and after and include the report in the pull request.

## Pull requests

Small, focused pull requests with a short description of the motivation. CI runs ruff, the tests on Python 3.10–3.13 and the scripted demo.
