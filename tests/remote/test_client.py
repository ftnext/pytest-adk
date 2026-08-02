# Copyright 2026 pytest-adk contributors

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from google.genai import types

from pytest_adk.remote.client import AdkApiClient

_BASE_URL = 'http://testserver'


def _client(handler, **kwargs: Any) -> AdkApiClient:
  """Build an AdkApiClient backed by an in-memory MockTransport."""
  return AdkApiClient(
      _BASE_URL, transport=httpx.MockTransport(handler), **kwargs
  )


async def test_list_apps_returns_app_names() -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.method == 'GET'
    assert request.url.path == '/list-apps'
    return httpx.Response(200, json=['weather_agent', 'home_automation_agent'])

  client = _client(handler)
  try:
    apps = await client.list_apps()
  finally:
    await client.aclose()

  assert apps == ['weather_agent', 'home_automation_agent']


async def test_create_session_sends_initial_state_and_parses_session() -> None:
  captured: dict[str, Any] = {}

  def handler(request: httpx.Request) -> httpx.Response:
    captured['method'] = request.method
    captured['path'] = request.url.path
    captured['body'] = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            'id': 'session-1',
            'appName': 'weather_agent',
            'userId': 'eval_user',
            'state': {'locale': 'en-US'},
            'events': [],
            'lastUpdateTime': 1_700_000_000.0,
        },
    )

  client = _client(handler)
  try:
    session = await client.create_session(
        app_name='weather_agent',
        user_id='eval_user',
        session_id='session-1',
        state={'locale': 'en-US'},
    )
  finally:
    await client.aclose()

  assert captured['method'] == 'POST'
  assert captured['path'] == '/apps/weather_agent/users/eval_user/sessions'
  # Request body uses the server's CreateSessionRequest camelCase alias.
  assert captured['body'] == {
      'sessionId': 'session-1',
      'state': {'locale': 'en-US'},
  }

  assert session.id == 'session-1'
  assert session.app_name == 'weather_agent'
  assert session.user_id == 'eval_user'
  assert session.state == {'locale': 'en-US'}


async def test_create_session_omits_unset_fields() -> None:
  captured: dict[str, Any] = {}

  def handler(request: httpx.Request) -> httpx.Response:
    captured['body'] = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            'id': 'generated-session',
            'appName': 'weather_agent',
            'userId': 'eval_user',
            'state': {},
            'events': [],
            'lastUpdateTime': 0.0,
        },
    )

  client = _client(handler)
  try:
    session = await client.create_session(
        app_name='weather_agent', user_id='eval_user'
    )
  finally:
    await client.aclose()

  assert captured['body'] == {}
  assert session.id == 'generated-session'


async def test_run_sends_camel_case_body() -> None:
  captured: dict[str, Any] = {}

  def handler(request: httpx.Request) -> httpx.Response:
    captured['path'] = request.url.path
    captured['body'] = json.loads(request.content)
    return httpx.Response(200, json=[])

  client = _client(handler)
  new_message = types.Content(
      role='user', parts=[types.Part(text='what is the weather?')]
  )
  try:
    events = await client.run(
        app_name='weather_agent',
        user_id='eval_user',
        session_id='session-1',
        new_message=new_message,
    )
  finally:
    await client.aclose()

  assert events == []
  assert captured['path'] == '/run'
  # RunAgentRequest uses common.BaseModel's to_camel alias generator; the
  # client must send the camelCase form, not snake_case.
  assert captured['body'] == {
      'appName': 'weather_agent',
      'userId': 'eval_user',
      'sessionId': 'session-1',
      'newMessage': {
          'role': 'user',
          'parts': [{'text': 'what is the weather?'}],
      },
  }


async def test_run_parses_events_and_ignores_unknown_fields() -> None:
  """Response events with unknown fields still validate (Event.extra='ignore').

  Also verifies a function_call part in the response round-trips onto the
  parsed Event, since tool-call/trajectory metrics depend on it.
  """

  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json=[
            {
                'invocationId': 'inv-1',
                'author': 'weather_agent',
                'id': 'event-1',
                'timestamp': 1_700_000_000.0,
                'content': {
                    'role': 'model',
                    'parts': [
                        {
                            'functionCall': {
                                'name': 'get_weather',
                                'args': {'city': 'Tokyo'},
                            }
                        }
                    ],
                },
                # Field that does not (yet) exist on the client's Event model.
                'someBrandNewServerField': {'nested': True},
            }
        ],
    )

  client = _client(handler)
  try:
    events = await client.run(
        app_name='weather_agent',
        user_id='eval_user',
        session_id='session-1',
        new_message=types.Content(role='user', parts=[types.Part(text='hi')]),
    )
  finally:
    await client.aclose()

  assert len(events) == 1
  event = events[0]
  assert event.invocation_id == 'inv-1'
  assert event.author == 'weather_agent'
  assert event.content is not None
  function_call = event.content.parts[0].function_call
  assert function_call is not None
  assert function_call.name == 'get_weather'
  assert function_call.args == {'city': 'Tokyo'}


async def test_delete_session_calls_expected_path() -> None:
  captured: dict[str, Any] = {}

  def handler(request: httpx.Request) -> httpx.Response:
    captured['method'] = request.method
    captured['path'] = request.url.path
    return httpx.Response(200)

  client = _client(handler)
  try:
    await client.delete_session(
        app_name='weather_agent', user_id='eval_user', session_id='session-1'
    )
  finally:
    await client.aclose()

  assert captured['method'] == 'DELETE'
  assert (
      captured['path']
      == '/apps/weather_agent/users/eval_user/sessions/session-1'
  )


@pytest.mark.parametrize(
    ('method_name', 'kwargs'),
    [
        ('list_apps', {}),
        (
            'create_session',
            {'app_name': 'weather_agent', 'user_id': 'eval_user'},
        ),
        (
            'run',
            {
                'app_name': 'weather_agent',
                'user_id': 'eval_user',
                'session_id': 'session-1',
                'new_message': types.Content(
                    role='user', parts=[types.Part(text='hi')]
                ),
            },
        ),
        (
            'delete_session',
            {
                'app_name': 'weather_agent',
                'user_id': 'eval_user',
                'session_id': 'session-1',
            },
        ),
    ],
)
async def test_404_raises_http_status_error(
    method_name: str, kwargs: dict[str, Any]
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={'detail': 'app not found'})

  client = _client(handler)
  try:
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
      await getattr(client, method_name)(**kwargs)
  finally:
    await client.aclose()

  assert excinfo.value.response.status_code == 404


async def test_500_raises_http_status_error() -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={'detail': 'internal error'})

  client = _client(handler)
  try:
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
      await client.run(
          app_name='weather_agent',
          user_id='eval_user',
          session_id='session-1',
          new_message=types.Content(
              role='user', parts=[types.Part(text='hi')]
          ),
      )
  finally:
    await client.aclose()

  assert excinfo.value.response.status_code == 500


async def test_custom_headers_are_sent_on_every_request() -> None:
  captured_headers: list[httpx.Headers] = []

  def handler(request: httpx.Request) -> httpx.Response:
    captured_headers.append(request.headers)
    return httpx.Response(200, json=['weather_agent'])

  client = _client(handler, headers={'Authorization': 'Bearer secret-token'})
  try:
    await client.list_apps()
  finally:
    await client.aclose()

  assert len(captured_headers) == 1
  assert captured_headers[0]['authorization'] == 'Bearer secret-token'


async def test_async_context_manager_closes_client() -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=[])

  async with _client(handler) as client:
    assert await client.list_apps() == []
  assert client._client.is_closed
