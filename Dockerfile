FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir ".[api,mcp,pdf]"

EXPOSE 8000
CMD ["refract", "serve", "--host", "0.0.0.0", "--port", "8000", "--data", "examples/demo"]
