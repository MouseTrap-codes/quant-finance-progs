ARG PY_VER=3.13.4
FROM python:${PY_VER}-slim-bullseye AS base

RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

FROM base AS builder

ENV POETRY_VERSION=1.8.3 POETRY_NO_INTERACTION=1 POETRY_VIRTUALENVS_CREATE=1 POETRY_VIRTUALENVS_IN_PROJECT=1

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && curl -sSL https://install.python-poetry.org | python3 - && ln -s /root/.local/bin/poetry /usr/local/bin/poetry && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root
COPY . .

FROM base AS runtime

RUN adduser --disabled-password --gecos "" quantuser
WORKDIR /app
COPY --from=builder /app /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER quantuser

CMD ["pytest", "-q"]
