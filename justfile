set default-list
set positional-arguments
set shell := ["bash", "--norc", "-euo", "pipefail", "-c"]

# Install the locked development environment with SSO support.
install *args:
    uv sync --locked --extra sso "$@"

# Run the regular test suite.
test *args:
    uv run --locked --extra sso pytest "$@"

# Run the PostgreSQL integration suite using TEST_CHAINLIT_DATABASE_URL.
test-postgres *args:
    uv run --locked --extra sso pytest -m integration tests/sso/test_postgres.py "$@"

# Check Just and Python formatting, then run Ruff.
lint:
    just --fmt --check
    uv run --locked ruff format --check src tests
    uv run --locked ruff check src tests

# Type-check the package source.
type-check:
    uv run --locked --extra sso ty check src

# Run all regular tests and static checks.
check: lint type-check test

# Format the Justfile and Python sources, then apply safe Ruff fixes.
format:
    just --fmt
    uv run --locked ruff format src tests
    uv run --locked ruff check src tests --fix --show-fixes

# Build fresh source and wheel distributions.
build *args:
    uv build --clear "$@"
