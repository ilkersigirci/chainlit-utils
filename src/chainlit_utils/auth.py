"""Authenticated-user helpers for Chainlit applications."""

import chainlit as cl


def authenticated_user_identifier() -> str:
    """Return the unique identifier of the current authenticated user."""
    user = cl.user_session.get("user")
    identifier = getattr(user, "identifier", None)
    if not isinstance(identifier, str) or not identifier:
        raise RuntimeError("Chainlit request has no authenticated user.")
    return identifier
