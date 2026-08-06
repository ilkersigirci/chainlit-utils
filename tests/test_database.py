from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chainlit_utils import database
from chainlit_utils.database import ChainlitMigrationError, Migration


class FakeTransaction:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    async def __aenter__(self) -> None:
        self.connection.transactions += 1

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakeConnection:
    def __init__(self, applied: dict[str, str] | None = None) -> None:
        self.applied = applied or {}
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self.transactions = 0
        self.closed = False

    async def execute(self, query: str, *args: object) -> str:
        self.executions.append((query, args))
        if "INSERT INTO" in query:
            version, checksum = args
            self.applied[str(version)] = str(checksum)
        return "OK"

    async def fetch(self, _query: str) -> list[dict[str, str]]:
        return [
            {"version": version, "checksum": checksum}
            for version, checksum in self.applied.items()
        ]

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def close(self) -> None:
        self.closed = True


def migration(version: str = "0001", checksum: str = "checksum") -> Migration:
    return Migration(version=version, checksum=checksum, sql="SELECT 42")


def test_load_migrations_orders_and_checksums_sql(tmp_path: Path) -> None:
    second = tmp_path / "0002_second.sql"
    second.write_text("SELECT 2;", encoding="utf-8")
    first = tmp_path / "0001_first.sql"
    first.write_text("SELECT 1;", encoding="utf-8")

    loaded = database.load_migrations(tmp_path)

    assert [item.version for item in loaded] == ["0001_first", "0002_second"]
    assert loaded[0].checksum == sha256(b"SELECT 1;").hexdigest()


def test_load_migrations_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ChainlitMigrationError, match="No Chainlit migrations"):
        database.load_migrations(tmp_path)


def test_bundled_migrations_remain_immutable() -> None:
    assert {item.version: item.checksum for item in database.load_migrations()} == {
        "0001_chainlit_data_layer": (
            "bff293c161ac5eb7787dae9edfdaca2fa4bd86c849eb909b6a3b1ae60bdb76e8"
        ),
        "0002_thread_tags": (
            "83ccf4ef72f89d1da5b37b82c9e91fa9b457fa89d53fdcb188670ecc97d94d5c"
        ),
        "0003_chainlit_2_1_0_command": (
            "f7858c37f38bde41f0770ac31f9d3a7cce7f2441018a61aa0971ed5c648455c1"
        ),
        "0004_chainlit_2_3_0_default_open": (
            "7337f8bdbab29d245012984b9e3d10e058cb7c452b8ead807cfbbed480148557"
        ),
        "0005_chainlit_2_9_4_modes": (
            "9474a6fa6581aeae0bca0d7c262dc0088d1332b5df06118fc1f289aae8d16b39"
        ),
    }


@pytest.mark.anyio
async def test_apply_migrations_applies_pending_scripts_atomically() -> None:
    connection = FakeConnection()

    await database.apply_migrations(connection, (migration(),))  # type: ignore[arg-type]

    assert connection.applied == {"0001": "checksum"}
    assert connection.transactions == 1
    assert any(query == "SELECT 42" for query, _ in connection.executions)
    assert connection.executions[-1] == (
        "SELECT pg_advisory_unlock($1)",
        (database.MIGRATION_LOCK_ID,),
    )


@pytest.mark.anyio
async def test_apply_migrations_skips_an_unchanged_version() -> None:
    connection = FakeConnection({"0001": "checksum"})

    await database.apply_migrations(connection, (migration(),))  # type: ignore[arg-type]

    assert connection.transactions == 0
    assert not any(query == "SELECT 42" for query, _ in connection.executions)


@pytest.mark.anyio
async def test_apply_migrations_rejects_drift_and_unlocks() -> None:
    connection = FakeConnection({"0001": "old-checksum"})

    with pytest.raises(ChainlitMigrationError, match="has changed"):
        await database.apply_migrations(  # type: ignore[arg-type]
            connection,
            (migration(),),
        )

    assert "pg_advisory_unlock" in connection.executions[-1][0]


@pytest.mark.anyio
async def test_apply_migrations_rejects_unknown_database_versions() -> None:
    connection = FakeConnection({"9999": "future"})

    with pytest.raises(ChainlitMigrationError, match=r"unknown.*9999"):
        await database.apply_migrations(  # type: ignore[arg-type]
            connection,
            (migration(),),
        )


@pytest.mark.anyio
async def test_migration_table_name_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        database.settings,
        "MIGRATIONS_TABLE",
        'history; DROP TABLE "Thread"',
    )

    with pytest.raises(ValueError, match="Invalid PostgreSQL"):
        await database.apply_migrations(  # type: ignore[arg-type]
            FakeConnection(),
            (migration(),),
        )


@pytest.mark.anyio
async def test_setup_closes_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = FakeConnection()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(database.asyncpg, "connect", connect)

    await database.setup_chainlit_schema("postgresql://example/database")

    connect.assert_awaited_once_with("postgresql://example/database")
    assert connection.closed


def test_cli_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(SystemExit, match="DATABASE_URL must be configured"):
        database.main()
