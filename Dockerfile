FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app

ENV UV_CACHE_DIR=/build-cache/uv

# Install project dependencies
RUN --mount=type=cache,target=/build-cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Install project
COPY . /app
RUN --mount=type=cache,target=/build-cache \
    uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

# Install NLTK data dependencies
RUN python -c "import nltk; nltk.download('punkt_tab')"
RUN python -c "import nltk; nltk.download('stopwords')"

# Install local models
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/qnli-electra-base')"

# Run application
EXPOSE 8080
CMD memmachine-sync-profile-schema && memmachine-server
