# syntax=docker/dockerfile:1.7
FROM python:3.12.10-slim

# ---- base env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ---- install Poetry 2.0.1 exactly ----
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    poetry --version  # should print 2.0.1

# ---- dependency layer (cache-friendly) ----
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-root --no-ansi

# ---- app code ----
# Copy only what you need; .dockerignore will exclude caches, .venv, etc.
COPY .streamlit /app/.streamlit
COPY core /app/core
COPY . /app

# Make Poetry venv CLI tools (streamlit, uvicorn, etc.) discoverable
ENV PATH="/app/.venv/bin:${PATH}"

# Add entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---- runtime ----
# Use a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
# Let the entrypoint map $PORT -> STREAMLIT_SERVER_PORT
ENTRYPOINT ["/app/entrypoint.sh"]