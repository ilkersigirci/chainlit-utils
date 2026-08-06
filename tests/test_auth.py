from types import SimpleNamespace

import pytest

from chainlit_utils import auth


class Session:
    def __init__(self, user: object = None) -> None:
        self.user = user

    def get(self, key: str, default: object = None) -> object:
        return self.user if key == "user" else default


def test_authenticated_user_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        auth.cl,
        "user_session",
        Session(SimpleNamespace(identifier="user-123")),
    )

    assert auth.authenticated_user_identifier() == "user-123"


@pytest.mark.parametrize("user", [None, SimpleNamespace(identifier="")])
def test_authenticated_user_identifier_requires_a_user(
    monkeypatch: pytest.MonkeyPatch,
    user: object,
) -> None:
    monkeypatch.setattr(auth.cl, "user_session", Session(user))

    with pytest.raises(RuntimeError, match="no authenticated user"):
        auth.authenticated_user_identifier()
