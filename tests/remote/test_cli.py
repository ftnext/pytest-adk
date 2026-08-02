# Copyright 2026 pytest-adk contributors
"""E2E tests for the ``pytest-adk eval`` CLI.

Calls :func:`pytest_adk.cli.main` directly (no subprocess) against the same
fake ``api_server`` used by ``test_eval_service.py`` (``tests/remote/fake_server.py``),
injected via ``main()``'s private ``transport`` hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from pytest_adk.cli import main

from .fake_server import FakeApiServer
from .fake_server import function_call_event
from .fake_server import text_event

_AGENT_URL = 'http://fake'
_APP_NAME = 'weather_agent'

_TEST_CONFIG_JSON = json.dumps({
    'criteria': {
        'tool_trajectory_avg_score': 1.0,
        'response_match_score': 0.8,
    }
})

_WEATHER_EVAL_SET_TOML = '''\
eval_set_id = "weather_set"

[[eval_cases]]
eval_id = "weather_case"

[[eval_cases.conversation]]
invocation_id = "inv-1"

[eval_cases.conversation.user_content]
role = "user"
parts = [ { text = "what is the weather in Tokyo?" } ]

[eval_cases.conversation.final_response]
role = "model"
parts = [ { text = "It is sunny in Tokyo." } ]

[eval_cases.conversation.intermediate_data]
tool_uses = [ { name = "get_weather", args = { city = "Tokyo" } } ]
'''


def _write_evalset(tmp_path: Path) -> Path:
  """Writes a minimal weather evalset + sibling test_config.json to tmp_path."""
  (tmp_path / 'test_config.json').write_text(_TEST_CONFIG_JSON, encoding='utf-8')
  evalset_path = tmp_path / 'weather.test.toml'
  evalset_path.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')
  return evalset_path


def _matching_script() -> list[list[dict]]:
  return [[
      function_call_event('', 'get_weather', {'city': 'Tokyo'}),
      text_event('', 'It is sunny in Tokyo.'),
  ]]


def _non_matching_script() -> list[list[dict]]:
  return [[
      function_call_event('', 'get_weather', {'city': 'Tokyo'}),
      text_event('', "I'm not sure, ask someone else."),
  ]]


def _saved_result_files(results_dir: Path, app_name: str = _APP_NAME) -> list[Path]:
  return list(
      (results_dir / app_name / '.adk' / 'eval_history').glob(
          '*.evalset_result.json'
      )
  )


def _transport_for(server: FakeApiServer) -> httpx.ASGITransport:
  return httpx.ASGITransport(app=server.build_app())


def test_eval_happy_path_saves_results_and_prints_location(
    tmp_path, capsys
) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  saved_files = _saved_result_files(results_dir)
  assert len(saved_files) == 1
  saved_result = json.loads(saved_files[0].read_text(encoding='utf-8'))
  assert saved_result['eval_set_id'] == 'weather_set'

  out = capsys.readouterr().out
  expected_dir = results_dir / _APP_NAME / '.adk' / 'eval_history'
  assert f'Eval results saved under: {expected_dir}' in out

  # Sessions created for this run were cleaned up (default keep_sessions=False).
  assert server.create_session_requests
  assert server.deleted_session_ids


def test_eval_metric_failure_exits_one_and_reports_per_case_metric(
    tmp_path, capsys
) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _non_matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 1
  # A result was still saved despite the failure (results are always visible
  # & persistent, per the design principle).
  assert len(_saved_result_files(results_dir)) == 1

  err = capsys.readouterr().err
  assert 'weather_case' in err
  assert 'response_match_score' in err


def test_app_name_omitted_resolves_when_exactly_one_app(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()  # default: a single app, _APP_NAME
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  assert len(_saved_result_files(results_dir, app_name=_APP_NAME)) == 1


def test_app_name_omitted_with_two_apps_exits_two_and_lists_candidates(
    tmp_path, capsys
) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer(app_names=['agent_one', 'agent_two'])
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'agent_one' in err
  assert 'agent_two' in err
  assert not results_dir.exists()


def test_fake_500_on_run_exits_two_and_reports_inference_failure(
    tmp_path, capsys
) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  server.fail_run_with_status = 500
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'INFERENCE FAILED' in err
  assert 'weather_case' in err
  # Nothing to evaluate/save: the only inference result was a FAILURE.
  assert not results_dir.exists()


def test_connection_error_exits_two(tmp_path) -> None:
  """A transport-level connection error surfaces as an inference FAILURE.

  ``--app-name`` is given explicitly, so ``GET /list-apps`` is never called;
  the connection error instead happens inside
  ``RemoteEvalService``'s session creation, which catches it per eval case
  (see ``eval_service.py``) rather than raising -- so this exercises the
  same "any inference FAILURE -> exit 2" path as the fake-500 test above,
  just via a lower-level transport failure.
  """
  evalset_path = _write_evalset(tmp_path)
  results_dir = tmp_path / 'results'

  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('connection refused', request=request)

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
      ],
      transport=httpx.MockTransport(handler),
  )

  assert exit_code == 2


def test_header_reaches_fake_server(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
          '--header',
          'Authorization: Bearer secret-token',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  assert server.received_headers
  assert all(
      headers.get('authorization') == 'Bearer secret-token'
      for headers in server.received_headers
  )


def test_malformed_header_is_an_argparse_error(tmp_path, capsys) -> None:
  evalset_path = _write_evalset(tmp_path)

  try:
    main([
        'eval',
        _AGENT_URL,
        str(evalset_path),
        '--header',
        'NoColonHere',
    ])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised
  assert 'NoColonHere' in capsys.readouterr().err


def test_invalid_agent_url_is_an_argparse_error(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)

  try:
    main(['eval', 'ftp://not-http', str(evalset_path)])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised


def test_keep_sessions_leaves_remote_sessions_undeleted(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
          '--keep-sessions',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  assert server.create_session_requests  # sessions WERE created
  assert server.deleted_session_ids == []  # but not deleted


def test_no_subcommand_prints_help_and_exits_two(capsys) -> None:
  exit_code = main([])

  assert exit_code == 2
  assert 'usage' in capsys.readouterr().out.lower()
