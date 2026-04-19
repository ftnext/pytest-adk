# Copyright 2026 pytest-adk contributors

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.agents.llm_agent import Agent

from pytest_adk import load_session_from_json
from pytest_adk import runner_from_exported_session

_FIXTURE = Path(__file__).parent / 'fixtures' / 'roll_die.json'


def test_load_roll_die_fixture_from_disk() -> None:
  session = load_session_from_json(_FIXTURE)
  assert session.app_name == 'fields_planner'
  assert session.user_id == 'user'
  assert session.id == '64f557af-53c3-4986-96c4-f1ca3e028cd2'
  assert len(session.events) == 12


@pytest.mark.asyncio
async def test_runner_from_roll_die_fixture_restores_events() -> None:
  agent = Agent(name='root', model='gemini-2.5-flash')
  runner, session = await runner_from_exported_session(agent, _FIXTURE)

  loaded = await runner.session_service.get_session(
      app_name=session.app_name,
      user_id=session.user_id,
      session_id=session.id,
  )
  assert loaded is not None
  assert len(loaded.events) == 12
  assert loaded.app_name == 'fields_planner'
