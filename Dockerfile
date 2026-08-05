FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "anthropic>=0.40.0" "numpy>=1.26.0" "rich>=13.0.0"

COPY noesis/ noesis/
COPY prism_bridge/ prism_bridge/
COPY prompts/ prompts/
COPY prism/ prism/
COPY examples/ examples/

ENV PYTHONPATH=/app

# Requires ANTHROPIC_API_KEY at runtime:
#   docker run -e ANTHROPIC_API_KEY=sk-ant-... ghcr.io/infinitule/noesis:latest
CMD ["python", "examples/basic_agent.py"]
