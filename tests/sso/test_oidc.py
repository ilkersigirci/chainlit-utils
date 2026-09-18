from unittest.mock import AsyncMock, Mock

import pytest

from chainlit_utils.sso.oidc import OidcClient, OidcConfig


def _config() -> OidcConfig:
    return OidcConfig(
        issuer="https://id.example",
        client_id="chainlit-client",
        client_secret="private-client-value",
        scopes="openid offline_access",
        resource="https://api.example/",
    )


async def test_metadata_accepts_the_configured_security_profile() -> None:
    client = OidcClient(_config())
    client.client.load_server_metadata = AsyncMock(
        return_value={
            "issuer": "https://id.example",
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic"],
        }
    )

    assert (await client.metadata())["issuer"] == "https://id.example"


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        (
            {
                "issuer": "https://other.example",
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["client_secret_basic"],
            },
            "issuer",
        ),
        (
            {
                "issuer": "https://id.example",
                "code_challenge_methods_supported": ["plain"],
                "token_endpoint_auth_methods_supported": ["client_secret_basic"],
            },
            "S256",
        ),
        (
            {
                "issuer": "https://id.example",
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["private_key_jwt"],
            },
            "client authentication method",
        ),
    ],
)
async def test_metadata_rejects_an_incompatible_security_profile(
    metadata: dict[str, object],
    error: str,
) -> None:
    client = OidcClient(_config())
    client.client.load_server_metadata = AsyncMock(return_value=metadata)

    with pytest.raises(ValueError, match=error):
        await client.metadata()


def test_injected_token_client_factory_receives_secret_configuration() -> None:
    token_client = Mock()
    factory = Mock(return_value=token_client)
    config = _config()
    client = OidcClient(config, token_client_factory=factory)

    assert client.token_client() is token_client
    factory.assert_called_once_with(config)
    assert "private-client-value" not in repr(config)
