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
import pytest

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


def test_zero_parallelism_is_an_argparse_error_and_does_not_hang(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)

  try:
    main(['eval', _AGENT_URL, str(evalset_path), '--parallelism', '0'])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised


def test_negative_parallelism_is_an_argparse_error(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)

  try:
    main(['eval', _AGENT_URL, str(evalset_path), '--parallelism', '-1'])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised


def test_zero_num_runs_is_an_argparse_error(tmp_path) -> None:
  evalset_path = _write_evalset(tmp_path)

  try:
    main(['eval', _AGENT_URL, str(evalset_path), '--num-runs', '0'])
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


def test_unparseable_agent_url_is_an_argparse_error(tmp_path, capsys) -> None:
  """A well-prefixed but unparseable URL must not reach httpx as a traceback.

  ``http://[::1`` (an unterminated IPv6 literal) makes ``httpx`` raise
  ``InvalidURL``; before it was validated at parse time, that escaped from
  ``AdkApiClient``'s construction -- which happens outside ``_run_eval``'s
  ``try`` -- as a traceback rather than the documented exit code.
  """
  evalset_path = _write_evalset(tmp_path)

  try:
    main(['eval', 'http://[::1', str(evalset_path)])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised
  assert 'not a valid URL' in capsys.readouterr().err


def test_directory_without_evalsets_exits_two(tmp_path, capsys) -> None:
  """An existing directory holding no *.test.* files must not exit 0."""
  evalset_dir = tmp_path / 'evals'
  evalset_dir.mkdir()
  # Misnamed: lacks the `.test.` infix, so directory discovery skips it.
  (evalset_dir / 'weather.toml').write_text(
      _WEATHER_EVAL_SET_TOML, encoding='utf-8'
  )
  server = FakeApiServer()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_dir),
          '--app-name',
          _APP_NAME,
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'No evalsets found' in err
  assert not results_dir.exists()
  # Nothing was ever sent to the remote server.
  assert server.run_requests == []


def test_duplicate_eval_set_id_exits_two(tmp_path, capsys) -> None:
  """Two files sharing an eval_set_id collide in InMemoryEvalSetsManager."""
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  first = tmp_path / 'one.test.toml'
  first.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')
  second = tmp_path / 'two.test.toml'
  # Same eval_set_id ("weather_set"), different eval case id.
  second.write_text(
      _WEATHER_EVAL_SET_TOML.replace('weather_case', 'weather_case_2'),
      encoding='utf-8',
  )
  server = FakeApiServer()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(first),
          str(second),
          '--app-name',
          _APP_NAME,
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'weather_set' in err
  assert 'Duplicate eval_set_id' in err
  assert not results_dir.exists()


def test_malformed_config_file_path_exits_two(tmp_path, capsys) -> None:
  evalset_path = _write_evalset(tmp_path)
  bad_config = tmp_path / 'broken_config.json'
  bad_config.write_text('{not valid json', encoding='utf-8')
  server = FakeApiServer()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--config-file-path',
          str(bad_config),
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  assert 'broken_config.json' in capsys.readouterr().err
  assert not results_dir.exists()


def test_non_json_list_apps_response_exits_two(tmp_path, capsys) -> None:
  """A URL pointing at a non-api_server must not raise a JSON decode error."""
  evalset_path = _write_evalset(tmp_path)
  results_dir = tmp_path / 'results'

  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, html='<html><body>Please sign in</body></html>'
    )

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--results-dir',
          str(results_dir),
      ],
      transport=httpx.MockTransport(handler),
  )

  assert exit_code == 2
  assert 'Failed to resolve --app-name' in capsys.readouterr().err
  assert not results_dir.exists()


def test_non_list_json_list_apps_response_exits_two(tmp_path, capsys) -> None:
  """Valid JSON of the wrong shape is an app-name resolution failure too."""
  evalset_path = _write_evalset(tmp_path)
  results_dir = tmp_path / 'results'

  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={'apps': ['weather_agent']})

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--results-dir',
          str(results_dir),
      ],
      transport=httpx.MockTransport(handler),
  )

  assert exit_code == 2
  assert 'Failed to resolve --app-name' in capsys.readouterr().err


_PROMPT_EVAL_SET_TOML = '''\
eval_set_id = "prompt_set"

[[eval_cases]]
eval_id = "prompt_case"

[[eval_cases.conversation]]
invocation_id = "inv-1"

[eval_cases.conversation.user_content]
role = "user"
parts = [ { text = "<prompt:prompt.txt DEVICE=lamp>" } ]

[eval_cases.conversation.final_response]
role = "model"
parts = [ { text = "Done." } ]
'''

_RESPONSE_ONLY_CONFIG_JSON = json.dumps(
    {'criteria': {'response_match_score': 0.8}}
)


def _write_prompt_evalset(tmp_path: Path, template_text: str) -> Path:
  """Writes an evalset whose only user turn is a <prompt:...> marker."""
  (tmp_path / 'test_config.json').write_text(
      _RESPONSE_ONLY_CONFIG_JSON, encoding='utf-8'
  )
  (tmp_path / 'prompt.txt').write_text(template_text, encoding='utf-8')
  evalset_path = tmp_path / 'prompt.test.toml'
  evalset_path.write_text(_PROMPT_EVAL_SET_TOML, encoding='utf-8')
  return evalset_path


def _sent_user_texts(server: FakeApiServer) -> list[str]:
  """The text of every newMessage the CLI actually sent to the fake server."""
  return [
      part['text']
      for request in server.run_requests
      for part in request['newMessage']['parts']
      if 'text' in part
  ]


def test_jinja_prompt_template_engine_renders_before_sending(tmp_path) -> None:
  """--prompt-template-engine jinja renders {{ VAR }} end-to-end."""
  evalset_path = _write_prompt_evalset(tmp_path, 'Turn on the {{ DEVICE }}.')
  server = FakeApiServer()
  server.scripts['cli_user'] = [[text_event('', 'Done.')]]
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
          '--prompt-template-engine',
          'jinja',
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  # The remote agent received the rendered prompt, not the marker or literal
  # Jinja syntax.
  assert _sent_user_texts(server) == ['Turn on the lamp.']
  assert len(_saved_result_files(results_dir)) == 1


def test_default_engine_leaves_jinja_placeholders_unrendered(tmp_path) -> None:
  """Contrast case: without the flag, {{ VAR }} is not expanded.

  This is exactly the silent-mismatch the flag exists to avoid -- the marker
  itself still expands (the prompt file is read), but ``string.Template``
  leaves ``{{ DEVICE }}`` alone, so the agent sees literal Jinja syntax.
  """
  evalset_path = _write_prompt_evalset(tmp_path, 'Turn on the {{ DEVICE }}.')
  server = FakeApiServer()
  server.scripts['cli_user'] = [[text_event('', 'Done.')]]
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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  assert _sent_user_texts(server) == ['Turn on the {{ DEVICE }}.']


def test_string_prompt_template_engine_renders_dollar_placeholders(
    tmp_path,
) -> None:
  """The explicit default keeps string.Template's ${VAR} working."""
  evalset_path = _write_prompt_evalset(tmp_path, 'Turn on the ${DEVICE}.')
  server = FakeApiServer()
  server.scripts['cli_user'] = [[text_event('', 'Done.')]]
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
          '--prompt-template-engine',
          'string',
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  assert _sent_user_texts(server) == ['Turn on the lamp.']


def test_invalid_prompt_template_engine_is_an_argparse_error(
    tmp_path, capsys
) -> None:
  evalset_path = _write_evalset(tmp_path)

  try:
    main([
        'eval',
        _AGENT_URL,
        str(evalset_path),
        '--prompt-template-engine',
        'mustache',
    ])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised
  assert 'mustache' in capsys.readouterr().err


def test_non_ascii_header_is_an_argparse_error(tmp_path, capsys) -> None:
  """httpx rejects non-ASCII headers when the client is built, outside the try."""
  evalset_path = _write_evalset(tmp_path)

  try:
    main([
        'eval',
        _AGENT_URL,
        str(evalset_path),
        '--header',
        'X-Label: café',
    ])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised
  assert 'ASCII' in capsys.readouterr().err


def test_path_like_app_name_from_server_is_rejected(tmp_path, capsys) -> None:
  """A remote-supplied '..' app name must not escape --results-dir.

  google-adk v2 rejects this itself, but v1 writes the result file outside the
  requested directory, so the CLI checks before persisting either way.
  """
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer(app_names=['..'])  # sole app -> auto-resolved
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'
  results_dir.mkdir()

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

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'Refusing to use app name' in err
  # Nothing was written anywhere -- neither inside nor outside results_dir.
  assert list(results_dir.rglob('*.json')) == []
  assert list(tmp_path.glob('.adk/**/*.json')) == []
  # Rejected before any inference was attempted.
  assert server.run_requests == []


def test_explicit_app_name_with_separator_is_rejected(tmp_path, capsys) -> None:
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          'a/../../b',
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  assert 'Refusing to use app name' in capsys.readouterr().err
  assert not results_dir.exists()


def test_unwritable_results_dir_exits_two(tmp_path, capsys) -> None:
  """A persistence failure must be an execution error, not exit 1.

  Exit 1 means "a metric failed"; automation would otherwise read a failed
  results write as a real evaluation verdict.
  """
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  # A *file* where the results directory should be: creating
  # {results_dir}/{app_name}/.adk/eval_history/ under it raises OSError.
  results_dir = tmp_path / 'results-as-a-file'
  results_dir.write_text('not a directory', encoding='utf-8')

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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  assert 'Evaluation failed' in capsys.readouterr().err


def test_one_empty_path_among_several_exits_two(tmp_path, capsys) -> None:
  """A populated path must not mask a sibling path that found nothing."""
  populated = tmp_path / 'populated'
  populated.mkdir()
  (populated / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  (populated / 'weather.test.toml').write_text(
      _WEATHER_EVAL_SET_TOML, encoding='utf-8'
  )
  empty = tmp_path / 'empty'
  empty.mkdir()
  # Misnamed, so directory discovery skips it.
  (empty / 'other.toml').write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')

  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(populated),
          str(empty),
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
  assert 'No evalsets found' in err
  assert 'empty' in err
  # Rejected before running the subset that *was* found.
  assert server.run_requests == []
  assert not results_dir.exists()


def test_evalset_with_no_eval_cases_exits_two(tmp_path, capsys) -> None:
  """A case-less evalset would score nothing yet report success."""
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'empty.test.toml'
  evalset_path.write_text(
      'eval_set_id = "empty_set"\neval_cases = []\n', encoding='utf-8'
  )
  server = FakeApiServer()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          _APP_NAME,
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'no eval cases' in err
  assert 'empty_set' in err
  assert server.run_requests == []
  assert not results_dir.exists()


def test_config_file_path_override_skips_broken_sibling_config(
    tmp_path, capsys
) -> None:
  """An explicit --config-file-path replaces sibling discovery entirely.

  The sibling test_config.json is malformed, which previously aborted the run
  inside _collect_eval_sets even though the caller supplied a valid config
  meant to take its place.
  """
  evalset_dir = tmp_path / 'evals'
  evalset_dir.mkdir()
  (evalset_dir / 'test_config.json').write_text(
      '{not valid json', encoding='utf-8'
  )
  evalset_path = evalset_dir / 'weather.test.toml'
  evalset_path.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')

  good_config = tmp_path / 'good_config.json'
  good_config.write_text(_TEST_CONFIG_JSON, encoding='utf-8')

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
          '--config-file-path',
          str(good_config),
          '--results-dir',
          str(results_dir),
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0, capsys.readouterr().err
  assert len(_saved_result_files(results_dir)) == 1


_REUSED_SESSION_EVAL_SET_TOML = '''\
eval_set_id = "reuse_set"

[[eval_cases]]
eval_id = "reuse_case"

[eval_cases.session_input]
app_name = "weather_agent"
user_id = "cli_user"
session_id = "pre-existing-session"

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


def _skip_without_session_id_support() -> None:
  """Skips unless this google-adk keeps an extra SessionInput.session_id.

  google-adk v2's SessionInput allows extra fields, so session_id survives
  loading; v1 forbids them, so such an evalset cannot even be loaded and the
  reuse feature does not exist there. Mirrors the probe in
  test_eval_service.py.
  """
  from google.adk.evaluation.eval_case import SessionInput

  probe = SessionInput.model_construct(app_name='a', user_id='b', session_id='c')
  if getattr(probe, 'session_id', None) != 'c':
    pytest.skip(
        "This google-adk version's SessionInput does not retain an extra"
        ' session_id field; existing-session reuse is not usable here.'
    )


def _write_reused_session_evalset(tmp_path: Path) -> Path:
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'reuse.test.toml'
  evalset_path.write_text(_REUSED_SESSION_EVAL_SET_TOML, encoding='utf-8')
  return evalset_path


def test_session_reuse_with_default_num_runs_exits_two(tmp_path, capsys) -> None:
  """Repeat runs would share one mutable remote session, so reject up front.

  --num-runs defaults to 2, so this is the *default* path for anyone using
  session reuse -- the reason it is an error rather than a warning.
  """
  _skip_without_session_id_support()
  evalset_path = _write_reused_session_evalset(tmp_path)
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
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'reuse_case' in err
  assert '--num-runs 1' in err
  # Rejected before any inference was attempted.
  assert server.run_requests == []
  assert not results_dir.exists()


def test_session_reuse_with_single_run_is_allowed(tmp_path, capsys) -> None:
  """--num-runs 1 has no repetition to contaminate, so reuse still works."""
  _skip_without_session_id_support()
  evalset_path = _write_reused_session_evalset(tmp_path)
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
          '--results-dir',
          str(results_dir),
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0, capsys.readouterr().err
  assert len(_saved_result_files(results_dir)) == 1
  # The pre-existing session was used as-is: never created, never deleted.
  assert server.create_session_requests == []
  assert server.deleted_session_ids == []
  assert all(
      req['sessionId'] == 'pre-existing-session' for req in server.run_requests
  )


def test_multiple_runs_without_session_reuse_are_unaffected(tmp_path) -> None:
  """The guard must not fire for ordinary evalsets that create sessions."""
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script() * 3
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
          '--num-runs',
          '3',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  # A fresh session per run, all cleaned up.
  assert len(server.create_session_requests) == 3
  assert len(server.deleted_session_ids) == 3
