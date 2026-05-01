FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade uv==0.11.7

COPY uv.lock pyproject.toml /app/

RUN uv sync --frozen --no-dev --no-install-project

COPY src /app/src

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "src.main:app"]
