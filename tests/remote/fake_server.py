# Copyright 2026 pytest-adk contributors
"""Shared fake ``adk api_server`` stand-in for remote-eval tests.

Extracted from ``test_eval_service.py`` (task R-2) so ``test_cli.py`` (task
R-3) can drive the same in-process fake server -- a small FastAPI app wired
via ``httpx.ASGITransport``, requiring no real network -- without duplicating
it. See ``REMOTE_EVAL_PLAN.md`` sections 6 (tasks R-2, R-3).
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from google.genai import types

from pytest_adk.remote.client import AdkApiClient

REMOTE_APP_NAME = 'weather_agent'  # default app name on the fake remote server
AGENT_AUTHOR = 'weather_agent'


def function_call_event(
    invocation_id: str, name: str, args: dict[str, Any]
) -> dict[str, Any]:
  return {
      'invocationId': invocation_id,
      'author': AGENT_AUTHOR,
      'content': {
          'role': 'model',
          'parts': [{'functionCall': {'name': name, 'args': args}}],
      },
  }


def text_event(invocation_id: str, text: str) -> dict[str, Any]:
  return {
      'invocationId': invocation_id,
      'author': AGENT_AUTHOR,
      'content': {'role': 'model', 'parts': [{'text': text}]},
  }


class FakeApiServer:
  """A minimal in-process stand-in for ``adk api_server``.

  Scripts each session's ``/run`` responses by the request's ``user_id``
  (each eval case in these tests uses a distinct ``session_input.user_id``,
  so this is a convenient, collision-free lookup key without smuggling
  test-only fields into session state).

  Session creation always returns a server-assigned id different from the
  one the client requested, so tests can verify callers use the *returned*
  id (not the requested one) for subsequent ``/run`` and ``DELETE`` calls --
  matching how a real api_server may ignore the requested id.
  """

  def __init__(self, app_names: list[str] | None = None) -> None:
    """Create a fake server.

    Args:
        app_names: Apps ``GET /list-apps`` reports. Defaults to a single app
            (``REMOTE_APP_NAME``); pass e.g. ``[]`` or two names to exercise
            ``--app-name`` auto-resolution failure paths.
    """
    self.app_names = [REMOTE_APP_NAME] if app_names is None else app_names
    self.create_session_requests: list[dict[str, Any]] = []
    self.run_requests: list[dict[str, Any]] = []
    self.deleted_session_ids: list[str] = []
    self._turn_counters: dict[str, int] = {}
    self._next_session_num = 0
    # user_id -> list of turns, each turn a list of event dicts (sans
    # invocationId, filled in per-call).
    self.scripts: dict[str, list[list[dict[str, Any]]]] = {}
    self.fail_user_ids: set[str] = set()
    self.fail_run_with_status: int | None = None
    # Headers of every request received, in arrival order (lower-cased keys,
    # matching httpx/Starlette's own normalization). Used by tests to verify
    # ``--header``/``headers=`` actually reach the server.
    self.received_headers: list[dict[str, str]] = []
    # The same headers as ``(name, value)`` pairs. ``dict(request.headers)``
    # keeps only the first value when a header name repeats, so tests about
    # repeated headers (e.g. two ``Cookie`` fields) need the raw pairs.
    self.received_header_pairs: list[list[tuple[str, str]]] = []

  def build_app(self) -> FastAPI:
    app = FastAPI()

    @app.get('/list-apps')
    async def list_apps(request: Request) -> list[str]:
      self.received_headers.append(dict(request.headers))
      self.received_header_pairs.append(
          [(k.decode(), v.decode()) for k, v in request.headers.raw]
      )
      return self.app_names

    @app.post('/apps/{app_name}/users/{user_id}/sessions')
    async def create_session(
        app_name: str, user_id: str, request: Request
    ) -> JSONResponse:
      self.received_headers.append(dict(request.headers))
      self.received_header_pairs.append(
          [(k.decode(), v.decode()) for k, v in request.headers.raw]
      )
      body = await request.json()
      self.create_session_requests.append(
          {'app_name': app_name, 'user_id': user_id, 'body': body}
      )
      self._next_session_num += 1
      # Deliberately NOT the client's requested sessionId (if any): see
      # class docstring.
      session_id = f'srv-session-{self._next_session_num}'
      return JSONResponse({
          'id': session_id,
          'appName': app_name,
          'userId': user_id,
          'state': body.get('state', {}),
          'events': [],
          'lastUpdateTime': 0.0,
      })

    @app.post('/run')
    async def run(request: Request):
      self.received_headers.append(dict(request.headers))
      self.received_header_pairs.append(
          [(k.decode(), v.decode()) for k, v in request.headers.raw]
      )
      body = await request.json()
      self.run_requests.append(body)
      user_id = body['userId']
      session_id = body['sessionId']

      if self.fail_run_with_status is not None:
        return JSONResponse(
            {'detail': 'boom'}, status_code=self.fail_run_with_status
        )
      if user_id in self.fail_user_ids:
        return JSONResponse({'detail': 'boom'}, status_code=500)

      turn_idx = self._turn_counters.get(session_id, 0)
      self._turn_counters[session_id] = turn_idx + 1

      script = self.scripts.get(user_id)
      if script is None or turn_idx >= len(script):
        return JSONResponse([])

      invocation_id = f'inv-{session_id}-{turn_idx}'
      events = [
          {**event, 'invocationId': invocation_id}
          for event in script[turn_idx]
      ]
      return JSONResponse(events)

    @app.delete('/apps/{app_name}/users/{user_id}/sessions/{session_id}')
    async def delete_session(
        app_name: str, user_id: str, session_id: str, request: Request
    ) -> Response:
      self.received_headers.append(dict(request.headers))
      self.received_header_pairs.append(
          [(k.decode(), v.decode()) for k, v in request.headers.raw]
      )
      self.deleted_session_ids.append(session_id)
      return Response(status_code=200)

    return app


def content(text: str) -> types.Content:
  return types.Content(role='user', parts=[types.Part(text=text)])


def client_for(server: FakeApiServer, **kwargs: Any) -> AdkApiClient:
  return AdkApiClient(
      'http://fake', transport=httpx.ASGITransport(app=server.build_app()), **kwargs
  )
