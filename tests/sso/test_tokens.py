from time import time
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet

from chainlit_utils.sso.oidc import OidcClient, OidcConfig
from chainlit_utils.sso.tokens import (
    OAuthLoginRequired,
    OAuthTokens,
    OAuthTokenStore,
)


def _oidc() -> OidcClient:
    return OidcClient(
        OidcConfig(
            issuer="https://id.example",
            client_id="client",
            client_secret="secret",
            scopes="openid",
        )
    )


def test_store_rejects_unsafe_table_names() -> None:
    with pytest.raises(ValueError, match="Invalid OAuth session table"):
        OAuthTokenStore(_oidc(), list, table_name="tokens; DROP TABLE users")


async def test_missing_or_undecryptable_grants_require_login() -> None:
    pool = AsyncMock()
    pool.fetchval.return_value = None
    key = Fernet.generate_key()

    async def pool_factory():
        return pool

    store = OAuthTokenStore(_oidc(), lambda: [key], pool_factory=pool_factory)
    with pytest.raises(OAuthLoginRequired, match="sign in again"):
        await store.access_token("missing", "alice")

    pool.fetchval.return_value = b"not-encrypted-by-this-key"
    with pytest.raises(OAuthLoginRequired):
        await store.access_token("corrupt", "alice")


def test_token_model_hides_secrets_and_requires_expiry() -> None:
    tokens = OAuthTokens(
        access_token="access-secret",
        refresh_token="refresh-secret",
        expires_at=time() + 60,
    )
    representation = repr(tokens)
    assert "access-secret" not in representation
    assert "refresh-secret" not in representation
