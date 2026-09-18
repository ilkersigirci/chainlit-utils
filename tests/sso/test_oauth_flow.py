"""End-to-end OAuth route tests with a local signing OIDC provider."""

from base64 import b64decode, urlsafe_b64encode
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlparse

import chainlit as cl
import httpx2
import jwt
import pytest
from authlib.integrations.httpx_client import AsyncOAuth2Client
from chainlit.auth import clear_auth_cookie
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request, Response
from jwt.algorithms import RSAAlgorithm
from starlette.responses import JSONResponse

from chainlit_utils.sso import chainlit as chainlit_oauth_module
from chainlit_utils.sso.chainlit import ChainlitOAuth
from chainlit_utils.sso.oidc import OidcClient, OidcConfig
from chainlit_utils.sso.tokens import OAuthLoginRequired, OAuthTokens


@dataclass
class OidcProvider:
    """Minimal signing provider for exercising Authlib's native flow."""

    auth_method: str = "client_secret_basic"
    resource: str | None = "https://api.example/"
    key: rsa.RSAPrivateKey = field(
        default_factory=lambda: rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
    )
    codes: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    claims_override: dict[str, object] = field(default_factory=dict)
    token_override: dict[str, object] = field(default_factory=dict)
    exchanges: int = 0
    revocations: list[str] = field(default_factory=list)
    revocation_status: int = 200
    invalid_signature: bool = False

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("openid-configuration"):
            return httpx2.Response(
                200,
                json={
                    "issuer": "https://id.example",
                    "authorization_endpoint": "https://id.example/authorize",
                    "token_endpoint": "https://id.example/token",
                    "jwks_uri": "https://id.example/jwks",
                    "revocation_endpoint": "https://id.example/revoke",
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": [self.auth_method],
                    "id_token_signing_alg_values_supported": ["RS256"],
                },
            )
        if request.url.path == "/jwks":
            key = RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
            return httpx2.Response(
                200,
                json={"keys": [{**key, "kid": "test-key", "use": "sig"}]},
            )

        form = parse_qs(request.content.decode(), keep_blank_values=True)
        if self.auth_method == "client_secret_basic":
            scheme, credentials = request.headers["authorization"].split()
            assert scheme == "Basic"
            assert b64decode(credentials).decode() == "chainlit-client:client-secret"
            assert "client_secret" not in form
        else:
            assert form["client_id"] == ["chainlit-client"]
            assert form["client_secret"] == ["client-secret"]
            assert "authorization" not in request.headers

        if request.url.path == "/revoke":
            self.revocations.append(form["token"][0])
            return httpx2.Response(self.revocation_status)

        assert request.url.path == "/token"
        self.exchanges += 1
        code = form["code"][0]
        params = self.codes.pop(code)
        assert form.get("resource") == ([self.resource] if self.resource else None)
        assert form["redirect_uri"] == [
            "https://chat.example/auth/oauth/generic/callback"
        ]
        challenge = (
            urlsafe_b64encode(sha256(form["code_verifier"][0].encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert params["code_challenge"] == [challenge]
        claims = {
            "iss": "https://id.example",
            "sub": "alice",
            "aud": "chainlit-client",
            "iat": int(time()),
            "exp": int(time()) + 3600,
            "nonce": params["nonce"][0],
            **self.claims_override,
        }
        signing_key = (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
            if self.invalid_signature
            else self.key
        )
        return httpx2.Response(
            200,
            json={
                "access_token": f"access-{code}",
                "refresh_token": f"refresh-{code}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "id_token": jwt.encode(
                    claims,
                    signing_key,
                    algorithm="RS256",
                    headers={"kid": "test-key"},
                ),
                **self.token_override,
            },
        )


@dataclass
class MemoryTokenStore:
    grants: dict[tuple[str, str], OAuthTokens] = field(default_factory=dict)
    browser_expirations: dict[tuple[str, str], float] = field(default_factory=dict)

    async def access_token(self, session_id: str, identifier: str) -> str:
        try:
            return self.grants[session_id, identifier].access_token
        except KeyError:
            raise OAuthLoginRequired() from None

    async def save(
        self,
        session_id: str,
        identifier: str,
        tokens: OAuthTokens,
        expires_at: float,
    ) -> None:
        key = session_id, identifier
        self.grants[key] = tokens
        self.browser_expirations[key] = expires_at

    async def delete(
        self,
        session_id: str,
        identifier: str,
    ) -> OAuthTokens | None:
        key = session_id, identifier
        self.browser_expirations.pop(key, None)
        return self.grants.pop(key, None)


@dataclass
class OAuthApp:
    app: FastAPI
    oidc: OidcClient
    provider: OidcProvider
    store: MemoryTokenStore
    persisted: dict[str, cl.User]

    def browser(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self.app),
            base_url="https://chat.example",
        )

    async def start(
        self,
        browser: httpx2.AsyncClient,
        code: str,
    ) -> dict[str, list[str]]:
        response = await browser.get("/auth/oauth/generic")
        assert response.status_code == 302
        params = parse_qs(
            urlparse(response.headers["location"]).query,
            keep_blank_values=True,
        )
        assert params["code_challenge_method"] == ["S256"]
        assert params.get("resource") == (
            [self.provider.resource] if self.provider.resource else None
        )
        self.provider.codes[code] = params
        return params

    async def login(
        self,
        browser: httpx2.AsyncClient,
        code: str,
    ) -> httpx2.Response:
        params = await self.start(browser, code)
        return await browser.get(
            "/auth/oauth/generic/callback",
            params={"state": params["state"][0], "code": code},
        )


def _token_client(
    config: OidcConfig,
    provider: OidcProvider,
) -> AsyncOAuth2Client:
    return AsyncOAuth2Client(
        client_id=config.client_id,
        client_secret=config.client_secret,
        token_endpoint_auth_method=config.client_auth_method,
        revocation_endpoint_auth_method=config.client_auth_method,
        timeout=config.timeout,
        trust_env=False,
        transport=httpx2.MockTransport(provider),
    )


@pytest.fixture
async def oauth_app(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> AsyncIterator[OAuthApp]:
    monkeypatch.setenv(
        "CHAINLIT_AUTH_SECRET",
        "test-signing-secret-with-at-least-32-bytes",
    )
    method, resource = getattr(
        request,
        "param",
        ("client_secret_basic", "https://api.example/"),
    )
    provider = OidcProvider(auth_method=method, resource=resource)
    config = OidcConfig(
        issuer="https://id.example",
        client_id="chainlit-client",
        client_secret="client-secret",
        scopes="openid offline_access api:invoke",
        resource=resource,
        client_auth_method=method,
    )
    oidc = OidcClient(
        config,
        token_client_factory=lambda value: _token_client(value, provider),
    )
    oidc.client.client_kwargs["transport"] = httpx2.MockTransport(provider)
    store = MemoryTokenStore()
    persisted: dict[str, cl.User] = {}

    async def create_user(user: cl.User) -> cl.User:
        persisted[user.identifier] = deepcopy(user)
        return persisted[user.identifier]

    layer = SimpleNamespace(create_user=AsyncMock(side_effect=create_user))
    monkeypatch.setattr(
        chainlit_oauth_module,
        "get_data_layer",
        Mock(return_value=layer),
    )
    monkeypatch.setattr(chainlit_oauth_module, "providers", [])
    monkeypatch.setattr(chainlit_oauth_module.cl, "oauth_callback", Mock())
    monkeypatch.setattr(
        chainlit_oauth_module.config.code,
        "password_auth_callback",
        chainlit_oauth_module.config.code.password_auth_callback,
    )
    monkeypatch.setattr(
        chainlit_oauth_module.config.code,
        "header_auth_callback",
        chainlit_oauth_module.config.code.header_auth_callback,
    )

    async def native_logout(request: Request, _response: Response) -> Response:
        response = JSONResponse({"success": True})
        clear_auth_cookie(request, response)
        return response

    monkeypatch.setattr(chainlit_oauth_module, "chainlit_logout", native_logout)
    oauth = ChainlitOAuth(
        provider_id="generic",
        chainlit_url="https://chat.example",
        auth_secret="test-signing-secret-with-at-least-32-bytes",
        oidc=oidc,
        provider_env=(),
        token_store=store,
        state_cookie="chainlit_utils_test_oauth_state",
    )
    app = FastAPI()
    oauth.configure(app)

    @app.exception_handler(OAuthLoginRequired)
    async def login_required(
        _request: Request,
        _error: OAuthLoginRequired,
    ) -> JSONResponse:
        return JSONResponse({"detail": "login required"}, status_code=401)

    @app.get("/credential")
    async def credential() -> str:
        return await oauth.credential()

    yield OAuthApp(app, oidc, provider, store, persisted)


@pytest.mark.parametrize(
    ("prompt", "provider_prompt", "expected"),
    [
        (None, None, None),
        ("login", None, "login"),
        ("login", "select_account", "select_account"),
    ],
)
async def test_authorization_uses_native_prompt_settings(
    oauth_app: OAuthApp,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str | None,
    provider_prompt: str | None,
    expected: str | None,
) -> None:
    for name, value in {
        "OAUTH_PROMPT": prompt,
        "OAUTH_GENERIC_PROMPT": provider_prompt,
    }.items():
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)

    async with oauth_app.browser() as browser:
        params = await oauth_app.start(browser, "unused")

    assert params.get("prompt") == ([expected] if expected is not None else None)


@pytest.mark.parametrize(
    "oauth_app",
    [
        ("client_secret_post", "https://api.example/"),
        ("client_secret_basic", None),
    ],
    indirect=True,
    ids=["post-with-resource", "basic-without-resource"],
)
async def test_pkce_login_keeps_grants_server_side_and_separates_sessions(
    oauth_app: OAuthApp,
) -> None:
    async with oauth_app.browser() as first, oauth_app.browser() as second:
        for browser, code in ((first, "first"), (second, "second")):
            response = await oauth_app.login(browser, code)
            assert response.status_code == 302
            assert "success=true" in response.headers["location"]
            auth_cookie = next(
                value
                for value in response.headers.get_list("set-cookie")
                if value.startswith("access_token=")
            )
            assert "HttpOnly" in auth_cookie
            assert "Secure" in auth_cookie
            assert "SameSite=lax" in auth_cookie
            assert "access-first" not in str(browser.cookies)
            assert "refresh-" not in str(browser.cookies)
            assert "chainlit_utils_test_oauth_state" not in browser.cookies

        first_jwt = first.cookies["access_token"]
        assert first_jwt != second.cookies["access_token"]
        for browser, code in ((first, "first"), (second, "second")):
            response = await browser.get("/credential")
            assert response.json() == f"access-{code}"

        assert (await first.post("/logout")).status_code == 200
        assert oauth_app.provider.revocations == ["refresh-first"]
        assert len(oauth_app.store.grants) == 1
        assert "access_token" not in first.cookies

        first.cookies.set("access_token", first_jwt)
        assert (await first.get("/credential")).status_code == 401
        assert (await second.get("/credential")).json() == "access-second"

    assert oauth_app.persisted["alice"].metadata == {"provider": "generic"}


async def test_callback_requires_a_validated_id_token(oauth_app: OAuthApp) -> None:
    def provider(request: httpx2.Request) -> httpx2.Response:
        response = oauth_app.provider(request)
        if request.url.path == "/token":
            token = response.json()
            token.pop("id_token")
            token["userinfo"] = {"sub": "unverified-subject"}
            return httpx2.Response(200, json=token)
        return response

    oauth_app.oidc.client.client_kwargs["transport"] = httpx2.MockTransport(provider)
    async with oauth_app.browser() as browser:
        response = await oauth_app.login(browser, "no-id-token")

        assert "error=" in response.headers["location"]
        assert "access_token" not in browser.cookies
    assert not oauth_app.store.grants
    assert not oauth_app.persisted


async def test_callback_keeps_authlibs_token_expiration(oauth_app: OAuthApp) -> None:
    expires_at = int(time()) + 120
    oauth_app.provider.token_override = {"expires_at": expires_at}

    async with oauth_app.browser() as browser:
        response = await oauth_app.login(browser, "expiry")

    assert "success=true" in response.headers["location"]
    stored = next(iter(oauth_app.store.grants.values()))
    assert stored.expires_at == expires_at
    assert all(value > time() for value in oauth_app.store.browser_expirations.values())


@pytest.mark.parametrize(
    "failure",
    ["state", "cookie", "nonce", "issuer", "audience", "expired", "signature"],
)
async def test_invalid_callback_does_not_create_a_session(
    oauth_app: OAuthApp,
    failure: str,
) -> None:
    async with oauth_app.browser() as browser:
        params = await oauth_app.start(browser, "invalid")
        state = params["state"][0]
        if failure == "state":
            state = "wrong-state"
        elif failure == "cookie":
            browser.cookies.clear()
        elif failure == "signature":
            oauth_app.provider.invalid_signature = True
        else:
            oauth_app.provider.claims_override = {
                "nonce": {"nonce": "wrong-nonce"},
                "issuer": {"iss": "https://evil.example"},
                "audience": {"aud": "another-client"},
                "expired": {"exp": int(time()) - 60},
            }[failure]
        response = await browser.get(
            "/auth/oauth/generic/callback",
            params={"code": "invalid", "state": state},
        )

        assert "error=" in response.headers["location"]
        assert "access_token" not in browser.cookies
        assert not oauth_app.store.grants
        if failure in ("state", "cookie"):
            assert oauth_app.provider.exchanges == 0


async def test_logout_stays_local_when_provider_revocation_fails(
    oauth_app: OAuthApp,
) -> None:
    oauth_app.provider.revocation_status = 503
    async with oauth_app.browser() as browser:
        await oauth_app.login(browser, "first")
        response = await browser.post("/logout")

        assert response.status_code == 200
        assert not oauth_app.store.grants
        assert "access_token" not in browser.cookies


async def test_logout_preserves_the_native_callback_response(
    oauth_app: OAuthApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def native_logout(request: Request, _response: Response) -> Response:
        assert not oauth_app.store.grants
        response = JSONResponse({"signed_out": True})
        response.delete_cookie("ui-preference")
        response.headers["X-Logout-Hook"] = "called"
        clear_auth_cookie(request, response)
        return response

    monkeypatch.setattr(chainlit_oauth_module, "chainlit_logout", native_logout)
    async with oauth_app.browser() as browser:
        await oauth_app.login(browser, "first")
        browser.cookies.set(
            "ui-preference",
            "compact",
            domain="chat.example",
            path="/",
        )
        response = await browser.post("/logout")

        assert response.status_code == 200
        assert response.json() == {"signed_out": True}
        assert response.headers["X-Logout-Hook"] == "called"
        assert "access_token" not in browser.cookies
        assert "ui-preference" not in browser.cookies
    assert oauth_app.provider.revocations == ["refresh-first"]


async def test_new_login_replaces_only_that_browser_session(
    oauth_app: OAuthApp,
) -> None:
    async with oauth_app.browser() as browser:
        await oauth_app.login(browser, "first")
        old_jwt = browser.cookies["access_token"]
        await oauth_app.login(browser, "second")

        assert len(oauth_app.store.grants) == 1
        stored = next(iter(oauth_app.store.grants.values()))
        assert stored.access_token == "access-second"
        assert (await browser.get("/credential")).json() == "access-second"

        browser.cookies.clear()
        browser.cookies.set("access_token", old_jwt)
        assert (await browser.get("/credential")).status_code == 401
