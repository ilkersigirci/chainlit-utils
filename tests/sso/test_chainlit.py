from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import anyio
import chainlit as cl
import httpx2
import pytest
from authlib.oidc.core import UserInfo
from chainlit.auth import create_jwt
from chainlit.context import ChainlitContext, context_var
from chainlit.session import WebsocketSession
from fastapi import FastAPI
from starlette.requests import Request

from chainlit_utils.sso.chainlit import (
    OAUTH_SESSION_CLAIM,
    ChainlitOAuth,
    OAuthRequestContextMiddleware,
    delegated_oauth_credential,
    session_identity,
)
from chainlit_utils.sso.oidc import OidcClient, OidcConfig
from chainlit_utils.sso.tokens import OAuthLoginRequired


@asynccontextmanager
async def chat_session(user: cl.User, token: str) -> AsyncIterator[WebsocketSession]:
    session = WebsocketSession(
        id=uuid4().hex,
        socket_id=uuid4().hex,
        emit=AsyncMock(),
        emit_call=AsyncMock(),
        user_env={},
        client_type="webapp",
        user=user,
        token=token,
    )
    context_token = context_var.set(ChainlitContext(session))
    try:
        yield session
    finally:
        context_var.reset(context_token)
        await session.delete()


@pytest.mark.parametrize("surface", ["http", "websocket"])
async def test_delegated_credentials_are_isolated_by_browser_login(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    requested: dict[str, str] = {}
    both_requested = anyio.Event()

    async def access_token(session_id: str, identifier: str) -> str:
        assert session_id == f"session-{identifier}"
        requested[identifier] = session_id
        if len(requested) == 2:
            both_requested.set()
        await both_requested.wait()
        return f"access-{identifier}"

    app = FastAPI()
    app.add_middleware(OAuthRequestContextMiddleware)

    @app.get("/credential")
    async def credential() -> str:
        return await delegated_oauth_credential(access_token)

    async def request_credential(identifier: str) -> None:
        user = cl.User(
            identifier=identifier,
            metadata={OAUTH_SESSION_CLAIM: f"session-{identifier}"},
        )
        token = create_jwt(user)
        if surface == "http":
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="https://chat.example",
            ) as browser:
                response = await browser.get(
                    "/credential",
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert response.json() == f"access-{identifier}"
        else:
            async with chat_session(user, token):
                assert await delegated_oauth_credential(access_token) == (
                    f"access-{identifier}"
                )

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(request_credential, "alice")
            group.start_soon(request_credential, "bob")
    assert requested == {
        "alice": "session-alice",
        "bob": "session-bob",
    }


def test_session_identity_rejects_unsigned_or_unbound_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    with pytest.raises(OAuthLoginRequired):
        session_identity(None)
    with pytest.raises(OAuthLoginRequired):
        session_identity("not-a-jwt")


async def test_websocket_user_must_match_the_signed_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    signed_user = cl.User(
        identifier="alice",
        metadata={OAUTH_SESSION_CLAIM: "session-alice"},
    )
    token = create_jwt(signed_user)
    access_token = AsyncMock(return_value="access-alice")

    async with chat_session(cl.User(identifier="bob"), token):
        with pytest.raises(OAuthLoginRequired):
            await delegated_oauth_credential(access_token)
    access_token.assert_not_awaited()


async def test_login_without_token_delegation_does_not_require_a_data_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET", "test-signing-secret-with-at-least-32-bytes"
    )
    oidc = OidcClient(
        OidcConfig(
            issuer="https://id.example",
            client_id="chainlit-client",
            client_secret="client-secret",
            scopes="openid",
        )
    )
    oidc.metadata = AsyncMock(return_value={})
    oidc.client.authorize_access_token = AsyncMock(
        return_value={"userinfo": UserInfo({"sub": "alice"})}
    )
    monkeypatch.setattr(
        "chainlit_utils.sso.chainlit.get_data_layer", Mock(return_value=None)
    )
    oauth = ChainlitOAuth(
        provider_id="generic",
        chainlit_url="https://chat.example",
        auth_secret="test-signing-secret-with-at-least-32-bytes",
        oidc=oidc,
        provider_env=(),
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/auth/oauth/generic/callback",
            "headers": [],
            "query_string": b"",
            "session": {},
        }
    )

    response = await oauth.callback("generic", request)

    assert response.status_code == 302
    assert response.headers["location"] == (
        "https://chat.example/login/callback?success=true"
    )
    assert "access_token=" in response.headers["set-cookie"]
