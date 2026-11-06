# Deployment: Docker, Compose, CI

## Docker

The image installs the wheel with the `api`, `mcp` and `pdf` extras. It runs as a non-root user (`refract`, uid 1000) and keeps the LLM cache and call log in the volume `/var/lib/refract`. By default it starts the REST API in demo mode, which needs no model and no key.

```bash
docker build -t refract .

# demo mode: recorded model answers on the packaged demo data
docker run --rm -p 8000:8000 refract
curl localhost:8000/health                       # … "demo": true …

# with a model: the same variables as .env (LLM_BASE_URL, LLM_API_KEY, REFRACT_MODEL, …)
docker run --rm -p 8000:8000 --env-file .env refract refract serve --host 0.0.0.0

# your own data folder (profiles/, events/, knowledge/)
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/my_data:/data:ro" -e REFRACT_DATA_DIR=/data \
  refract refract serve --host 0.0.0.0

# a config file
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/refract.toml:/config/refract.toml:ro" \
  -e REFRACT_CONFIG=/config/refract.toml refract refract serve --host 0.0.0.0

# any other command: the interactive console, one analysis, the MCP server over stdio
docker run --rm -it --env-file .env refract refract shell
docker run --rm --env-file .env refract refract analyze --profile demo_coffee_ip --event ev_usn_vat_ru
docker run --rm -i refract refract mcp --demo
```

On Windows PowerShell, write `${PWD}` instead of `$PWD`. A model on the host (for example Ollama) is reachable from the container at `http://host.docker.internal:11434/v1`. On Linux, add `--add-host=host.docker.internal:host-gateway`.

Set `REFRACT_API_TOKEN` to require `Authorization: Bearer <token>` on every endpoint except `/health`. The health check calls `/health` every 30 seconds.

## Docker Compose with local models

`docker-compose.yml` runs three services:

- `ollama` is the model server. It sets `OLLAMA_CONTEXT_LENGTH=16384` because agent prompts are 6–12k tokens.
- `pull` runs once to download the models, then exits.
- `refract` is the API. It uses `examples/config/ollama-local.toml`, with endpoints and models taken from the environment.

```bash
docker compose up -d                              # the first start downloads ≈ 9 GB of models
docker compose run --rm refract refract check     # structured output, context window, reasoning
docker compose run --rm refract refract shell
MAIN_MODEL=qwen3.5:4b VERIFIER_MODEL=gemma4:e4b docker compose up -d   # smaller models
```

For an NVIDIA GPU, uncomment the `deploy` block of `ollama` (this needs the NVIDIA Container Toolkit). Without a GPU the models run on the CPU, which is slow: expect minutes per analysis. The comments in `examples/config/ollama-local.toml` list model sizes by VRAM.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:

| Job | What it checks |
| --- | --- |
| `lint` | `ruff check`, `ruff format --check`, `pyright src` |
| `test` | `pytest` on Linux (Python 3.10–3.13), Windows and macOS (3.10, 3.13); then the CLI without a model: `refract demo`, `analyze --demo`, `brief --demo`, a scripted `refract shell` session |
| `package` | `uv build`; the wheel installed alone in a clean environment on Linux, Windows and macOS, run from outside the source tree, so the packaged demo data and recordings are checked |
| `docker` | the image builds, `docker compose config` is valid, the API in demo mode answers `/health` and `POST /insights` |

None of the jobs needs a key or a model.
