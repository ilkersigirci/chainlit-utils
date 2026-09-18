"""Encrypted, expiring OAuth grants stored in Chainlit's PostgreSQL pool."""

import re
from collections.abc import Awaitable, Callable, Sequence
from time import time

import asyncpg
from authlib.integrations.base_client import OAuthError
from chainlit.data import get_data_layer
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chainlit_utils.sso.oidc import OidcClient
from openai import OpenAIError

PoolFactory = Callable[[], Awaitable[asyncpg.Pool]]
EncryptionKeys = Callable[[], Sequence[str | bytes]]


class OAuthTokens(BaseModel):
    """The provider credentials retained for one browser login."""

    model_config = ConfigDict(hide_input_in_errors=True)

    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str | None = Field(default=None, min_length=1, repr=False)
    expires_at: float = Field(gt=0, allow_inf_nan=False)


class OAuthLoginRequired(OpenAIError):
    """Signal that an OpenAI request cannot obtain a delegated credential."""

    def __init__(self) -> None:
        super().__init__("Authorization expired. Log out and sign in again.")


async def chainlit_postgres_pool() -> asyncpg.Pool:
    """Return the pool owned by Chainlit's official PostgreSQL data layer."""
    layer = get_data_layer()
    if not isinstance(layer, ChainlitDataLayer):
        raise RuntimeError("OAuth token forwarding requires Chainlit PostgreSQL.")
    await layer.connect()
    assert layer.pool is not None
    return layer.pool


class OAuthTokenStore:
    """Persist and refresh encrypted OAuth grants for independent browser logins."""

    def __init__(
        self,
        oidc: OidcClient,
        encryption_keys: EncryptionKeys,
        *,
        pool_factory: PoolFactory = chainlit_postgres_pool,
        table_name: str = "chainlit_utils_oauth_sessions",
    ) -> None:
        self.oidc = oidc
        self.encryption_keys = encryption_keys
        self.pool_factory = pool_factory
        self.table = _quoted_identifier(table_name)
        self.lock_name = f"chainlit_utils.sso.tokens.{table_name}"

    async def initialize(self) -> None:
        """Create storage once and delete expired browser sessions."""
        pool = await self.pool_factory()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))", self.lock_name
            )
            await connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    session_id TEXT PRIMARY KEY,
                    identifier TEXT NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL,
                    tokens BYTEA NOT NULL
                )
            """)
            await connection.execute(
                f"DELETE FROM {self.table} WHERE expires_at <= $1", time()
            )

    async def save(
        self,
        session_id: str,
        identifier: str,
        tokens: OAuthTokens,
        expires_at: float,
    ) -> None:
        """Save one encrypted grant after removing expired sessions."""
        pool = await self.pool_factory()
        await pool.execute(f"DELETE FROM {self.table} WHERE expires_at <= $1", time())
        await pool.execute(
            f"INSERT INTO {self.table} (session_id, identifier, expires_at, tokens) VALUES ($1, $2, $3, $4)",
            session_id,
            identifier,
            expires_at,
            self._cipher().encrypt(tokens.model_dump_json().encode()),
        )

    async def delete(self, session_id: str, identifier: str) -> OAuthTokens | None:
        """Commit local invalidation before provider revocation is attempted."""
        pool = await self.pool_factory()
        encrypted = await pool.fetchval(
            f"DELETE FROM {self.table} WHERE session_id = $1 AND identifier = $2 RETURNING tokens",
            session_id,
            identifier,
        )
        try:
            return self._decrypt(encrypted)
        except OAuthLoginRequired:
            return None

    async def access_token(self, session_id: str, identifier: str) -> str:
        """Return a usable access token, refreshing it once across all workers."""
        pool = await self.pool_factory()
        query = (
            f"SELECT tokens FROM {self.table} "
            "WHERE session_id = $1 AND identifier = $2 AND expires_at > $3"
        )
        tokens = self._decrypt(
            await pool.fetchval(query, session_id, identifier, time())
        )
        if tokens.expires_at > time() + 30:
            return tokens.access_token

        async with pool.acquire() as connection, connection.transaction():
            await connection.execute("SET LOCAL lock_timeout = '10s'")
            tokens = self._decrypt(
                await connection.fetchval(
                    query + " FOR UPDATE", session_id, identifier, time()
                )
            )
            if tokens.expires_at > time() + 30:
                return tokens.access_token
            if tokens.refresh_token is None:
                raise OAuthLoginRequired()
            metadata = await self.oidc.metadata()
            try:
                async with self.oidc.token_client() as client:
                    result = await client.refresh_token(
                        metadata["token_endpoint"],
                        refresh_token=tokens.refresh_token,
                        resource=self.oidc.config.resource,
                    )
            except OAuthError as exc:
                if exc.error in ("invalid_grant", "invalid_token"):
                    raise OAuthLoginRequired() from None
                raise
            refreshed = OAuthTokens.model_validate(result)
            if refreshed.refresh_token is None:
                refreshed.refresh_token = tokens.refresh_token
            await connection.execute(
                f"UPDATE {self.table} SET tokens = $2 WHERE session_id = $1",
                session_id,
                self._cipher().encrypt(refreshed.model_dump_json().encode()),
            )
            return refreshed.access_token

    def _cipher(self) -> MultiFernet:
        keys = self.encryption_keys()
        if not keys:
            raise RuntimeError("At least one OAuth encryption key is required.")
        return MultiFernet([Fernet(key) for key in keys])

    def _decrypt(self, encrypted: bytes | None) -> OAuthTokens:
        if encrypted is None:
            raise OAuthLoginRequired()
        try:
            return OAuthTokens.model_validate_json(self._cipher().decrypt(encrypted))
        except (InvalidToken, ValidationError):
            raise OAuthLoginRequired() from None


def _quoted_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ValueError(f"Invalid OAuth session table name: {value!r}.")
    return f'"{value}"'


__all__ = [
    "OAuthLoginRequired",
    "OAuthTokenStore",
    "OAuthTokens",
    "chainlit_postgres_pool",
]
