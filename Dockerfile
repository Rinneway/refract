# Refract in a container: the REST API by default, any `refract …` command otherwise.
#
#   docker build -t refract .
#   docker run --rm -p 8000:8000 refract                                   # demo mode: no model, recorded answers
#   docker run --rm -p 8000:8000 --env-file .env refract refract serve --host 0.0.0.0
#   docker run --rm -i refract refract mcp --demo                          # MCP over stdio
#
# With your own data: -v "$PWD/my_data:/data" -e REFRACT_DATA_DIR=/data. With local models: docker-compose.yml.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir "hatchling>=1.26" build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
RUN python -m build --wheel --no-isolation --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LLM_CACHE_DIR=/var/lib/refract/cache LLM_LOG_PATH=/var/lib/refract/logs/llm_calls.jsonl
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[api,mcp,pdf]" && rm /tmp/*.whl \
    && useradd --create-home --uid 1000 refract && mkdir -p /var/lib/refract && chown refract /var/lib/refract
USER refract
WORKDIR /home/refract
VOLUME ["/var/lib/refract"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["refract", "serve", "--host", "0.0.0.0", "--port", "8000", "--demo"]
