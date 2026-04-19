# Copyright 2026 pytest-adk contributors

from __future__ import annotations

import pytest
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types

from pytest_adk import load_session_from_json
from pytest_adk import runner_from_exported_session


def test_load_session_from_json_file(tmp_path) -> None:
  session = Session(
      id='sid',
      app_name='app1',
      user_id='uid1',
      state={'plain': 1},
      events=[],
  )
  path = tmp_path / 'export.session.json'
  path.write_text(session.model_dump_json(), encoding='utf-8')
  loaded = load_session_from_json(path)
  assert loaded.id == 'sid'
  assert loaded.app_name == 'app1'
  assert loaded.user_id == 'uid1'
  assert loaded.state['plain'] == 1


@pytest.mark.asyncio
async def test_runner_from_exported_session_restores_via_session_service() -> None:
  exported = Session(
      id='sess-1',
      app_name='myapp',
      user_id='user-1',
      state={'counter': 2},
      events=[
          Event(
              invocation_id='inv-1',
              author='user',
              content=types.Content(
                  role='user',
                  parts=[types.Part(text='hello')],
              ),
          ),
      ],
  )
  agent = Agent(name='root', model='gemini-2.5-flash')

  runner, session = await runner_from_exported_session(agent, exported)

  assert session.app_name == 'myapp'
  assert session.user_id == 'user-1'
  assert session.id == 'sess-1'

  loaded = await runner.session_service.get_session(
      app_name='myapp',
      user_id='user-1',
      session_id='sess-1',
  )
  assert loaded is not None
  assert len(loaded.events) == 1
  assert loaded.events[0].author == 'user'
  assert loaded.state.get('counter') == 2


@pytest.mark.asyncio
async def test_runner_from_exported_session_accepts_json_string() -> None:
  exported = Session(
      id='s-json',
      app_name='app-json',
      user_id='u-json',
      state={},
      events=[
          Event(
              invocation_id='inv-json',
              author='user',
              content=types.Content(
                  role='user',
                  parts=[types.Part(text='ping')],
              ),
          ),
      ],
  )
  agent = Agent(name='root', model='gemini-2.5-flash')
  runner, _ = await runner_from_exported_session(agent, exported.model_dump_json())

  loaded = await runner.session_service.get_session(
      app_name='app-json',
      user_id='u-json',
      session_id='s-json',
  )
  assert loaded is not None
  assert len(loaded.events) == 1
