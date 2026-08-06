"""Versioned PostgreSQL migrations for Chainlit conversation persistence."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import asyncpg

from chainlit_utils.settings import settings

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_LOCK_ID = 0x434C5554494C5301

logger = logging.getLogger(__name__)


class ChainlitMigrationError(RuntimeError):
    """Raised when Chainlit migrations are missing, unknown, or changed."""


@dataclass(frozen=True)
class Migration:
    """A checksum-protected SQL migration."""

    version: str
    checksum: str
    sql: str


def load_migrations(directory: Path | None = None) -> tuple[Migration, ...]:
    """Load the bundled SQL migrations in version order."""
    selected_directory = MIGRATIONS_DIR if directory is None else directory
    migrations = []
    for path in sorted(selected_directory.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=path.stem,
                checksum=sha256(sql.encode()).hexdigest(),
                sql=sql,
            )
        )
    if not migrations:
        raise ChainlitMigrationError(
            f"No Chainlit migrations found in {selected_directory}."
        )
    return tuple(migrations)


async def apply_migrations(
    connection: asyncpg.Connection,
    migrations: tuple[Migration, ...] | None = None,
) -> None:
    """Apply each pending migration atomically and reject migration drift."""
    quoted_table = _quoted_identifier(settings.MIGRATIONS_TABLE)
    await connection.execute("SELECT pg_advisory_lock($1)", MIGRATION_LOCK_ID)
    try:
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quoted_table} (
                version TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        rows = await connection.fetch(f"SELECT version, checksum FROM {quoted_table}")
        applied = {row["version"]: row["checksum"] for row in rows}
        selected_migrations = load_migrations() if migrations is None else migrations
        known_versions = {migration.version for migration in selected_migrations}
        unknown_versions = sorted(set(applied) - known_versions)
        if unknown_versions:
            joined_versions = ", ".join(unknown_versions)
            raise ChainlitMigrationError(
                "Database contains Chainlit migrations unknown to this release: "
                f"{joined_versions}."
            )

        for migration in selected_migrations:
            applied_checksum = applied.get(migration.version)
            if applied_checksum == migration.checksum:
                continue
            if applied_checksum is not None:
                raise ChainlitMigrationError(
                    f"Applied Chainlit migration {migration.version!r} has changed."
                )

            async with connection.transaction():
                # Bundled migrations are trusted assets and may contain multiple SQL
                # statements. asyncpg executes the complete script as one command.
                await connection.execute(migration.sql)
                await connection.execute(
                    f"""
                    INSERT INTO {quoted_table} (version, checksum)
                    VALUES ($1, $2)
                    """,
                    migration.version,
                    migration.checksum,
                )
    finally:
        await connection.execute("SELECT pg_advisory_unlock($1)", MIGRATION_LOCK_ID)


async def setup_chainlit_schema(
    database_url: str,
) -> None:
    """Initialize or migrate Chainlit's PostgreSQL schema."""
    connection = await asyncpg.connect(database_url)
    try:
        await apply_migrations(connection)
    finally:
        await connection.close()


def main() -> None:
    """Apply pending migrations using the native ``DATABASE_URL`` setting."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL must be configured.")
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing PostgreSQL Chainlit persistence schema")
    asyncio.run(setup_chainlit_schema(database_url))
    logger.info("PostgreSQL Chainlit persistence schema is ready")


def _quoted_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ValueError(f"Invalid PostgreSQL migration table name: {value!r}.")
    return f'"{value}"'


if __name__ == "__main__":
    main()
