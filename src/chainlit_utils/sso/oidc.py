"""Configured Authlib clients for PKCE-protected OpenID Connect login."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuth, StarletteOAuth2App

OAuthClientAuthMethod = Literal["client_secret_basic", "client_secret_post"]


@dataclass(frozen=True)
class OidcConfig:
    """Identity-provider settings needed by the reusable OIDC integration."""

    issuer: str
    client_id: str
    client_secret: str = field(repr=False)
    scopes: str
    resource: str | None = None
    client_auth_method: OAuthClientAuthMethod = "client_secret_basic"
    timeout: float = 10


class OidcClient:
    """Own Authlib discovery, validation, token exchange, and refresh clients."""

    def __init__(
        self,
        config: OidcConfig,
        *,
        token_client_factory: Callable[[OidcConfig], AsyncOAuth2Client] | None = None,
    ) -> None:
        self.config = config
        self._token_client_factory = token_client_factory
        client = OAuth().register(
            "oidc",
            client_id=config.client_id,
            client_secret=config.client_secret,
            server_metadata_url=(
                f"{config.issuer.rstrip('/')}/.well-known/openid-configuration"
            ),
            client_kwargs={
                **self._client_kwargs(),
                "scope": config.scopes,
                "code_challenge_method": "S256",
            },
        )
        assert isinstance(client, StarletteOAuth2App)
        self.client = client

    async def metadata(self) -> dict[str, Any]:
        """Load discovery metadata and enforce the configured security profile."""
        metadata = await self.client.load_server_metadata()
        if metadata.get("issuer") != self.config.issuer:
            raise ValueError(
                "OIDC discovery issuer does not match the configured issuer."
            )
        if "S256" not in metadata.get("code_challenge_methods_supported", []):
            raise ValueError("OIDC provider must advertise S256 PKCE support.")
        if self.config.client_auth_method not in metadata.get(
            "token_endpoint_auth_methods_supported", ["client_secret_basic"]
        ):
            raise ValueError(
                "OIDC provider does not support the configured client authentication method."
            )
        return metadata

    def token_client(self) -> AsyncOAuth2Client:
        """Build an OAuth client for refresh and revocation requests."""
        if self._token_client_factory is not None:
            return self._token_client_factory(self.config)
        return AsyncOAuth2Client(
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            **self._client_kwargs(),
        )

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "token_endpoint_auth_method": self.config.client_auth_method,
            "revocation_endpoint_auth_method": self.config.client_auth_method,
            "timeout": self.config.timeout,
            "trust_env": False,
        }


__all__ = ["OAuthClientAuthMethod", "OidcClient", "OidcConfig"]
