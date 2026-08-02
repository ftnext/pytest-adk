# Copyright 2026 pytest-adk contributors
"""Thin async client for the ADK ``api_server`` REST API.

Import constraint: this module must stay importable without google-adk's
``eval`` extra (no pandas/tabulate). Only import from ``google.adk.events``,
``google.adk.sessions``, ``google.genai`` and ``httpx`` here -- never from
``google.adk.evaluation``.

Verified against google-adk 2.5.0
(``.venv/lib/python3.13/site-packages/google/adk/cli/api_server.py``):

- ``GET /list-apps`` returns a plain ``list[str]`` of app names.
- ``POST /apps/{app_name}/users/{user_id}/sessions`` creates a session. The
  request body is ``CreateSessionRequest`` (``session_id``, ``state``,
  ``events``, all optional); the response body is a ``Session``.
- ``POST /run`` runs one turn and returns the raw ``list[Event]`` produced by
  the agent. The request body is ``RunAgentRequest`` (``app_name``,
  ``user_id``, ``session_id``, ``new_message``, plus optional fields this
  client does not use).
- ``DELETE /apps/{app_name}/users/{user_id}/sessions/{session_id}`` deletes a
  session and returns no body.

Both ``RunAgentRequest`` and ``CreateSessionRequest`` extend
``google.adk.cli.utils.common.BaseModel``, whose ``model_config`` sets
``alias_generator=to_camel`` and ``populate_by_name=True``: the server accepts
(and, for response models such as ``Session``/``Event``, emits) camelCase
keys. Requests are therefore built directly as camelCase dicts here, so this
client does not need to import ``google.adk.cli`` (which pulls in FastAPI) --
it only ever *speaks* the same wire format.
"""

from __future__ import annotations

import urllib.parse
from types import TracebackType
from typing import Any
from typing import Sequence

import httpx
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types


def _quote(path_segment: str, *, field: str) -> str:
  """Percent-encode one dynamic REST path segment.

  ``safe=''`` because these values are single path segments: httpx escapes
  some characters when normalizing a URL (a space becomes ``%20``) but leaves
  reserved ones alone, and silently resolves ``..`` against the preceding
  segment -- so the encoding has to happen here.

  Dots are *unreserved* in RFC 3986, so ``quote()`` alone leaves ``'.'`` and
  ``'..'`` intact and the dot-segment removal in URL resolution would still
  drop or rewrite them. An all-dots segment therefore gets its dots
  percent-encoded explicitly; ``%2E`` is equivalent to ``.`` for a server
  decoding the segment, but is no longer a dot-segment during resolution.

  A literal ``/`` is rejected outright rather than encoded. ``%2F`` does not
  help: an ADK ``api_server`` is a FastAPI/ASGI app whose router matches the
  *decoded* ``scope['path']``, so the encoded slash turns back into a
  separator before ``/apps/{app_name}/users/{user_id}/sessions`` is matched
  and the request 404s. There is no way to carry a slash in one of these path
  parameters, so a clear error beats a mystifying 404 mid-run. (Verified
  against a real FastAPI router: every other character tried -- ``.``,
  ``..``, spaces, ``?``, ``#``, ``\\``, even a literal ``%2F`` -- round-trips
  to the server unchanged; only ``/`` breaks.)

  Args:
      path_segment: The value to place in one path segment.
      field: Name of the field the value came from, for the error message.

  Raises:
      ValueError: If ``path_segment`` contains ``/``.
  """
  if '/' in path_segment:
    raise ValueError(
        f'{field} {path_segment!r} must not contain "/": the api_server'
        ' routes these as single path segments, and an encoded slash is'
        ' decoded back into a separator before routing, so such a value'
        ' cannot be addressed at all.'
    )
  quoted = urllib.parse.quote(path_segment, safe='')
  if quoted and set(quoted) == {'.'}:
    return quoted.replace('.', '%2E')
  return quoted


class AdkApiClient:
  """Thin async client for the ADK ``api_server`` REST API.

  Wraps :class:`httpx.AsyncClient` and knows only the handful of REST paths
  and request/response shapes needed to drive a remote ADK agent for
  evaluation: listing apps, creating/deleting sessions, and running a turn.

  This client performs no retries. ``/run`` is not idempotent, so a failed
  request is surfaced to the caller via :meth:`httpx.Response.raise_for_status`
  rather than retried transparently.
  """

  def __init__(
      self,
      base_url: str,
      *,
      headers: dict[str, str] | Sequence[tuple[str, str]] | None = None,
      timeout: float = 300.0,
      transport: httpx.AsyncBaseTransport | None = None,
  ) -> None:
    """Create a client bound to a running ``adk api_server`` instance.

    Args:
        base_url: Base URL of the api_server, e.g. ``http://localhost:8000``.
        headers: Extra HTTP headers sent with every request, e.g.
            ``{'Authorization': 'Bearer ...'}``. A sequence of ``(name,
            value)`` pairs is also accepted, and is what the CLI passes, so
            that a header name may legitimately repeat.
        timeout: Per-request timeout in seconds. Agent inference can be slow,
            hence the generous default.
        transport: Optional ``httpx`` transport, forwarded to
            ``httpx.AsyncClient``. Tests inject ``httpx.MockTransport`` here
            to exercise this client without a real network call.
    """
    self._client = httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        transport=transport,
    )

  async def __aenter__(self) -> AdkApiClient:
    """Return ``self`` so the client can be used as an async context manager."""
    return self

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc: BaseException | None,
      traceback: TracebackType | None,
  ) -> None:
    """Close the underlying HTTP client on context-manager exit."""
    await self.aclose()

  async def aclose(self) -> None:
    """Close the underlying HTTP client."""
    await self._client.aclose()

  async def list_apps(self) -> list[str]:
    """List the app names the server can run.

    Returns:
        App names as reported by ``GET /list-apps``.

    Raises:
        httpx.HTTPStatusError: If the server responds with a 4xx/5xx status.
        ValueError: If the response body is not JSON, or is not a JSON list of
            strings. A URL that points at something other than an api_server
            (or a redirect to an HTML login page) typically lands here rather
            than on a 4xx/5xx status.
    """
    response = await self._client.get('/list-apps')
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not all(
        isinstance(app, str) for app in payload
    ):
      raise ValueError(
          'GET /list-apps did not return a JSON list of app names, got'
          f' {payload!r}.'
      )
    return payload

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str | None = None,
      state: dict[str, Any] | None = None,
  ) -> Session:
    """Create a session, optionally with an explicit ID and/or initial state.

    Args:
        app_name: Name of the app to create the session under.
        user_id: User the session belongs to.
        session_id: Explicit session ID. If omitted, the server generates one.
        state: Initial session state.

    Returns:
        The created :class:`~google.adk.sessions.Session`.

    Raises:
        httpx.HTTPStatusError: If the server responds with a 4xx/5xx status.
    """
    body: dict[str, Any] = {}
    if session_id is not None:
      body['sessionId'] = session_id
    if state is not None:
      body['state'] = state
    response = await self._client.post(
        f'/apps/{_quote(app_name, field="app_name")}'
        f'/users/{_quote(user_id, field="user_id")}/sessions',
        json=body,
    )
    response.raise_for_status()
    return Session.model_validate(response.json())

  async def run(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      new_message: types.Content,
  ) -> list[Event]:
    """Run one turn of the agent and return the raw ADK events it produced.

    Args:
        app_name: Name of the app to run.
        user_id: User the session belongs to.
        session_id: Existing session ID to run the turn in.
        new_message: The user message for this turn.

    Returns:
        The events the agent produced, parsed with ``Event.model_validate``.
        ``Event.model_config`` sets ``extra='ignore'``, so fields added by a
        newer server-side ADK version are dropped rather than raising.

    Raises:
        httpx.HTTPStatusError: If the server responds with a 4xx/5xx status.
    """
    body = {
        'appName': app_name,
        'userId': user_id,
        'sessionId': session_id,
        'newMessage': new_message.model_dump(
            mode='json', by_alias=True, exclude_none=True
        ),
    }
    response = await self._client.post('/run', json=body)
    response.raise_for_status()
    return [Event.model_validate(event) for event in response.json()]

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Delete a session.

    Args:
        app_name: Name of the app the session belongs to.
        user_id: User the session belongs to.
        session_id: Session ID to delete.

    Raises:
        httpx.HTTPStatusError: If the server responds with a 4xx/5xx status.
    """
    response = await self._client.delete(
        f'/apps/{_quote(app_name, field="app_name")}'
        f'/users/{_quote(user_id, field="user_id")}'
        f'/sessions/{_quote(session_id, field="session_id")}'
    )
    response.raise_for_status()
