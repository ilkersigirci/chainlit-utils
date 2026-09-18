"""Chainlit login routes and request-scoped OAuth token delegation."""

import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from time import time
from typing import Protocol, cast
from uuid import uuid4

import chainlit as cl
import httpx2
from authlib.common.errors import AuthlibBaseError
from authlib.oidc.core import UserInfo
from chainlit.auth import clear_auth_cookie, create_jwt
from chainlit.auth.cookie import OAuth2PasswordBearerWithCookie
from chainlit.auth.jwt import decode_jwt
from chainlit.config import config
from chainlit.context import ChainlitContextException
from chainlit.data import get_data_layer
from chainlit.oauth_providers import OAuthProvider, providers
from chainlit.server import logout as chainlit_logout
from chainlit.session import WebsocketSession
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from joserfc.errors import JoseError
from jwt import PyJWTError
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from chainlit_utils.sso.oidc import OidcClient
from chainlit_utils.sso.tokens import (
    OAuthLoginRequired,
    OAuthTokens,
)

logger = logging.getLogger(__name__)
OAUTH_SESSION_CLAIM = "chainlit_utils.oauth_session"

_request_token: ContextVar[str | None] = ContextVar(
    "chainlit_utils_oauth_request_token", default=None
)
_browser_auth = OAuth2PasswordBearerWithCookie(tokenUrl="/login", auto_error=False)


class OAuthTokenStorage(Protocol):
    async def access_token(self, session_id: str, identifier: str) -> str: ...

    async def save(
        self,
        session_id: str,
        identifier: str,
        tokens: OAuthTokens,
        expires_at: float,
    ) -> None: ...

    async def delete(
        self,
        session_id: str,
        identifier: str,
    ) -> OAuthTokens | None: ...


def session_identity(
    token: str | None,
    *,
    session_claim: str = OAUTH_SESSION_CLAIM,
) -> tuple[str, str]:
    """Read session identity only from Chainlit's verified browser JWT."""
    if not token:
        raise OAuthLoginRequired()
    try:
        user = decode_jwt(token)
    except (PyJWTError, JoseError, ValueError, TypeError):
        raise OAuthLoginRequired() from None
    session_id = user.metadata.get(session_claim)
    if not isinstance(session_id, str) or not session_id:
        raise OAuthLoginRequired()
    return session_id, user.identifier


class OAuthRequestContextMiddleware:
    """Make the current HTTP browser token available to credential callbacks."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        context_token = _request_token.set(await _browser_auth(Request(scope)))
        try:
            await self.app(scope, receive, send)
        finally:
            _request_token.reset(context_token)


async def delegated_oauth_credential(
    access_token: Callable[[str, str], Awaitable[str]],
    *,
    session_claim: str = OAUTH_SESSION_CLAIM,
) -> str:
    """Resolve and verify the current browser login before loading its grant."""
    token = _request_token.get()
    try:
        session = cl.context.session
    except ChainlitContextException:
        session = None
    if isinstance(session, WebsocketSession):
        token = session.token
    session_id, identifier = session_identity(token, session_claim=session_claim)
    if isinstance(session, WebsocketSession) and (
        session.user is None or session.user.identifier != identifier
    ):
        raise OAuthLoginRequired()
    return await access_token(session_id, identifier)


class OidcLoginButton(OAuthProvider):
    """Advertise an OIDC provider while the parent app owns its secure routes."""

    def __init__(self, provider_id: str, required_env: Sequence[str]) -> None:
        self.id = provider_id
        self.env = list(required_env)


class ChainlitOAuth:
    """Own OIDC login, logout, and per-browser downstream token delegation."""

    def __init__(
        self,
        *,
        provider_id: str,
        chainlit_url: str,
        auth_secret: str,
        oidc: OidcClient,
        provider_env: Sequence[str],
        token_store: OAuthTokenStorage | None = None,
        session_claim: str = OAUTH_SESSION_CLAIM,
        state_cookie: str = "chainlit_utils_oauth_state",
    ) -> None:
        self.provider_id = provider_id
        self.chainlit_url = chainlit_url.rstrip("/")
        self.auth_secret = auth_secret
        self.oidc = oidc
        self.token_store = token_store
        self.session_claim = session_claim
        self.state_cookie = state_cookie
        self.login_button = OidcLoginButton(provider_id, provider_env)
        self.router = APIRouter()
        self.router.add_api_route(
            "/auth/oauth/{provider_id}",
            self.authorize,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/auth/oauth/{provider_id}/callback",
            self.callback,
            methods=["GET"],
        )
        self.router.add_api_route("/logout", self.logout, methods=["POST"])

    async def credential(self) -> str:
        """Return the delegated token for the current HTTP or WebSocket session."""
        if self.token_store is None:
            raise RuntimeError("OAuth token delegation is not enabled.")
        return await delegated_oauth_credential(
            self.token_store.access_token,
            session_claim=self.session_claim,
        )

    async def authorize(self, provider_id: str, request: Request):
        """Begin authorization with PKCE and optional resource targeting."""
        self._check_provider(provider_id)
        await self.oidc.metadata()
        return await self.oidc.client.authorize_redirect(
            request,
            f"{self.chainlit_url}/auth/oauth/{provider_id}/callback",
            resource=self.oidc.config.resource,
            prompt=self.login_button.get_prompt(),
        )

    async def callback(self, provider_id: str, request: Request):
        """Validate the ID token, persist any delegated grant, and log in."""
        self._check_provider(provider_id)
        try:
            await self.oidc.metadata()
            result = await self.oidc.client.authorize_access_token(
                request,
                resource=self.oidc.config.resource,
                leeway=10,
            )
            tokens = (
                OAuthTokens.model_validate(result)
                if self.token_store is not None
                else None
            )
            identity = result.get("userinfo")
            if not isinstance(identity, UserInfo):
                raise ValueError("Missing validated ID token.")
            subject = identity.get("sub")
            if not isinstance(subject, str) or not subject:
                raise ValueError("Missing verified subject.")
        except (httpx2.HTTPError, AuthlibBaseError, JoseError, ValueError):
            request.session.clear()
            return RedirectResponse(
                f"{self.chainlit_url}/login?error=oauth_callback_error",
                status_code=302,
            )

        user = cl.User(identifier=subject, metadata={"provider": provider_id})
        if layer := get_data_layer():
            await layer.create_user(user)
        if tokens is not None:
            assert self.token_store is not None
            session_id = uuid4().hex
            await self.token_store.save(
                session_id,
                subject,
                tokens,
                time() + config.project.user_session_timeout,
            )
            try:
                previous_identity = session_identity(
                    await _browser_auth(request),
                    session_claim=self.session_claim,
                )
            except OAuthLoginRequired:
                pass
            else:
                await self.token_store.delete(*previous_identity)
            user.metadata[self.session_claim] = session_id

        response = RedirectResponse(
            f"{self.chainlit_url}/login/callback?success=true",
            status_code=302,
        )
        clear_auth_cookie(request, response)
        response.set_cookie(
            os.environ.get("CHAINLIT_AUTH_COOKIE_NAME", "access_token"),
            create_jwt(user),
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=config.project.user_session_timeout,
        )
        request.session.clear()
        return response

    async def logout(self, request: Request, response: Response):
        """Delete the local grant, try provider revocation, then log out."""
        tokens: OAuthTokens | None = None
        if self.token_store is not None:
            try:
                identity = session_identity(
                    await _browser_auth(request),
                    session_claim=self.session_claim,
                )
            except OAuthLoginRequired:
                pass
            else:
                tokens = await self.token_store.delete(*identity)
        request.session.clear()
        if tokens is not None:
            try:
                metadata = await self.oidc.metadata()
                if endpoint := metadata.get("revocation_endpoint"):
                    async with self.oidc.token_client() as client:
                        revoked = await client.revoke_token(
                            endpoint,
                            token=tokens.refresh_token or tokens.access_token,
                            token_type_hint=(
                                "refresh_token"
                                if tokens.refresh_token
                                else "access_token"
                            ),
                        )
                        revoked.raise_for_status()
            except (httpx2.HTTPError, AuthlibBaseError, ValueError):
                logger.warning(
                    "OAuth provider revocation failed; local delegated grant was removed."
                )
        return await chainlit_logout(request, response)

    def configure(self, app: FastAPI) -> None:
        """Register the secure OAuth callbacks before Chainlit is mounted."""
        config.code.password_auth_callback = None
        config.code.header_auth_callback = None
        cast(list[OAuthProvider], providers)[:] = [self.login_button]
        cl.oauth_callback(self._reject_native_callback)
        app.include_router(self.router)
        if self.token_store is not None:
            app.add_middleware(OAuthRequestContextMiddleware)
        app.add_middleware(
            SessionMiddleware,
            secret_key=self.auth_secret,
            session_cookie=self.state_cookie,
            max_age=300,
            https_only=True,
            same_site="lax",
        )

    async def _reject_native_callback(self, *_args: object) -> None:
        raise OAuthLoginRequired()

    def _check_provider(self, provider_id: str) -> None:
        if provider_id != self.provider_id:
            raise HTTPException(404, "Unknown OAuth provider.")


__all__ = [
    "OAUTH_SESSION_CLAIM",
    "ChainlitOAuth",
    "OAuthRequestContextMiddleware",
    "OAuthTokenStorage",
    "OidcLoginButton",
    "delegated_oauth_credential",
    "session_identity",
]
