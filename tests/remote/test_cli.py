# Copyright 2026 pytest-adk contributors
"""E2E tests for the ``pytest-adk eval`` CLI.

Calls :func:`pytest_adk.cli.main` directly (no subprocess) against the same
fake ``api_server`` used by ``test_eval_service.py`` (``tests/remote/fake_server.py``),
injected via ``main()``'s private ``transport`` hook.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import warnings
from pathlib import Path

import httpx
import pytest

import pytest_adk.cli as cli_module
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


def _write_evalset(tmp_path: Path, config_json: str = _TEST_CONFIG_JSON) -> Path:
  """Writes a minimal weather evalset + sibling test_config.json to tmp_path."""
  tmp_path.mkdir(parents=True, exist_ok=True)
  (tmp_path / 'test_config.json').write_text(config_json, encoding='utf-8')
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


# --- google-adk warning suppression ------------------------------------------
#
# `pytest-adk eval` imports `google.adk.evaluation.local_eval_service` (via the
# lazily-imported `RemoteEvalService`), which is noisy: a `vertexai.preview.rag`
# deprecation `UserWarning`, plus several `[EXPERIMENTAL] ...` `UserWarning`s
# from ADK's `@experimental` decorator. `_silence_google_adk_warnings` (see
# cli.py) filters these out by default without touching warnings that are not
# google-adk's to begin with.


def test_google_adk_warnings_are_silenced_but_user_warnings_are_not() -> None:
  """Unit test of the helper: no ADK import needed, so this is version-independent.

  ``warnings.resetwarnings()`` clears whatever ambient filters this process
  (or an earlier test) has accumulated, giving a deterministic starting point:
  with an empty filter list, the two filters the helper appends are the only
  ones in play, so a warning that matches neither of them falls through to
  Python's implicit default action (show once) rather than a leftover 'always'
  or 'ignore' entry deciding its fate instead.
  """
  with warnings.catch_warnings(record=True) as records:
    warnings.resetwarnings()
    cli_module._silence_google_adk_warnings()
    # Mirrors the real vertexai.preview.rag deprecation warning: no
    # `[EXPERIMENTAL]` prefix, attributed to a `google.adk` submodule.
    warnings.warn_explicit(
        'The `vertexai.preview.rag` module is deprecated',
        UserWarning,
        'vertexai.py',
        19,
        module='google.adk.dependencies.vertexai',
        registry={},
    )
    # Mirrors an [EXPERIMENTAL] warning attributed to pytest_adk's own module
    # (stacklevel=2 from inside LocalEvalService.__init__), which the
    # `module=` filter alone would not catch.
    warnings.warn('[EXPERIMENTAL] LocalEvalService: ...', UserWarning)
    # A genuine warning from the user's own code -- must not be touched.
    warnings.warn('my agent said something', UserWarning)

  assert [str(r.message) for r in records] == ['my agent said something']


def test_pythonwarnings_style_filter_overrides_the_silencing_filters() -> None:
  """Pins the documented opt-out: ``PYTHONWARNINGS=always::UserWarning``.

  ``-W`` / ``PYTHONWARNINGS`` filters are installed by the interpreter at
  startup, i.e. they are already in ``warnings.filters`` before ``main()``
  runs. ``simplefilter`` reproduces exactly that starting state here
  (it inserts at the *front* of the list, as startup filters effectively
  are), which is why this can be asserted in-process instead of paying for
  a subprocess. ``_silence_google_adk_warnings``' ``append=True`` then puts
  its filters *behind* the user's, so the user's matches first and both
  google-adk warnings stay visible.
  """
  with warnings.catch_warnings(record=True) as records:
    warnings.resetwarnings()
    warnings.simplefilter('always', UserWarning)
    cli_module._silence_google_adk_warnings()
    warnings.warn('[EXPERIMENTAL] LocalEvalService: ...', UserWarning)
    warnings.warn_explicit(
        'The `vertexai.preview.rag` module is deprecated',
        UserWarning,
        'vertexai.py',
        19,
        module='google.adk.dependencies.vertexai',
        registry={},
    )

  assert [str(r.message) for r in records] == [
      '[EXPERIMENTAL] LocalEvalService: ...',
      'The `vertexai.preview.rag` module is deprecated',
  ]


def test_importing_pytest_adk_installs_no_warning_filters() -> None:
  """The silencing filters belong to the CLI, not to ``import pytest_adk``.

  This is the invariant behind the ``PLUGGABLE_AUTH`` limitation documented
  in ``_silence_google_adk_warnings`` and the README: that warning fires
  while ``google.adk`` is imported, so the only way to catch it would be a
  filter installed at package import time -- which would silence warnings
  for the pytest plugin path and for every library consumer too, not just
  ``pytest-adk eval``. Deliberately not done, and pinned here.

  Asserts the absence of *these* filters specifically rather than that
  ``warnings.filters`` is unchanged: importing pytest-adk pulls in authlib
  and urllib3, which legitimately register filters of their own
  (``AuthlibDeprecationWarning``, ``DependencyWarning``, ...), so a
  length/equality check against a bare interpreter would fail for reasons
  that have nothing to do with pytest-adk.

  Needs a subprocess: ``pytest_adk`` is already imported in this process, so
  an in-process ``import`` would be a no-op and prove nothing.
  """
  probe = (
      'import warnings, json\n'
      'import pytest_adk\n'
      'print(json.dumps([\n'
      '    [f[0], getattr(f[1], "pattern", None), f[2].__name__,\n'
      '     getattr(f[3], "pattern", None)]\n'
      '    for f in warnings.filters\n'
      ']))\n'
  )
  completed = subprocess.run(
      [sys.executable, '-c', probe], capture_output=True, text=True, check=True
  )
  filters = json.loads(completed.stdout)

  silencing = [
      f
      for f in filters
      if f[0] == 'ignore'
      and f[2] == 'UserWarning'
      and (f[1] == r'\[EXPERIMENTAL\]' or f[3] == r'google\.adk(\.|$)')
  ]
  assert silencing == [], (
      'importing pytest_adk installed the CLI\'s google-adk silencing '
      f'filters ({silencing}); they must stay confined to cli.main() so the '
      'pytest plugin path and library consumers keep their warnings'
  )


def test_an_experimental_prefixed_user_warning_is_suppressed_too() -> None:
  """Pins the documented exception to "your own warnings are unaffected".

  The ``[EXPERIMENTAL]`` rule cannot also key on a module (ADK's
  ``@experimental`` warns with ``stacklevel=2``, so the warning is
  attributed to pytest-adk's own calling module, not to ``google.adk``), so
  it necessarily matches on message text alone -- and therefore also hides a
  user warning that happens to start with the same prefix. That trade-off is
  documented in cli.py's module docstring and in the README's Limitations
  section; this test exists so the behavior and those docs cannot drift
  apart silently.
  """
  with warnings.catch_warnings(record=True) as records:
    warnings.resetwarnings()
    cli_module._silence_google_adk_warnings()
    warnings.warn('[EXPERIMENTAL] my own metric API', UserWarning)
    # Same custom metric, message not starting with the prefix: still visible.
    warnings.warn('my own metric API is experimental', UserWarning)

  assert [str(r.message) for r in records] == [
      'my own metric API is experimental'
  ]


def test_main_does_not_leak_warning_filters(tmp_path, capsys) -> None:
  """Pins the ``catch_warnings()`` wrapper around ``main()``'s body.

  Without it, each in-process ``main()`` call would append two more entries
  to the global ``warnings.filters`` list (``filterwarnings(message=...)``
  compiles a fresh regex each time, so repeats never compare equal and
  never coalesce), growing it unboundedly across the test suite's ~40 calls.
  """
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'
  filters_snapshot = list(warnings.filters)

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

  assert exit_code == 0, capsys.readouterr().err
  assert warnings.filters == filters_snapshot


def test_main_silences_adk_experimental_warnings(tmp_path, capsys) -> None:
  """No `[EXPERIMENTAL]` notice reaches the CLI's actual output.

  This checks `capsys`-captured stdout/stderr rather than pytest's `recwarn`
  fixture: `recwarn` installs its own catch-all filter (`simplefilter`, which
  -- despite the name -- inserts at the *front* of `warnings.filters`) so
  that it can record every warning regardless of category or message: that
  filter always matches first, before the `append=True` filters
  `_silence_google_adk_warnings` adds, so it structurally cannot observe
  anything being suppressed. Checking the captured output instead exercises
  the behavior a real CLI invocation (with no such fixture already holding a
  catch-all filter) actually has.
  """
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

  captured = capsys.readouterr()
  assert exit_code == 0, captured.err
  assert '[EXPERIMENTAL]' not in captured.out
  assert '[EXPERIMENTAL]' not in captured.err


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
  captured = capsys.readouterr()
  assert 'INFERENCE FAILED' in captured.err
  assert 'weather_case' in captured.err
  # Nothing to evaluate/save: the only inference result was a FAILURE.
  assert not results_dir.exists()
  # So the run must not advertise a results path that was never written --
  # it says so explicitly instead.
  assert 'Eval results saved under' not in captured.out
  assert 'No eval results were saved' in captured.err


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


def test_empty_session_id_is_not_a_pinned_session(tmp_path, capsys) -> None:
  """`session_id = ""` means "not pinned", consistently in both layers.

  TOML has no null literal, so an empty string is the only way to spell "no
  pinned session" for a field that would otherwise be omitted. The guard and
  the runtime share one predicate, so this must neither be rejected by the
  --num-runs guard nor take the runtime's reuse path (which would run the
  conversation against a session literally named '' and never clean it up).
  """
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'reuse.test.toml'
  evalset_path.write_text(
      _REUSED_SESSION_EVAL_SET_TOML.replace(
          'session_id = "pre-existing-session"', 'session_id = ""'
      ),
      encoding='utf-8',
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script() * 2
  results_dir = tmp_path / 'results'

  # Default --num-runs (2): not rejected by the guard.
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

  assert exit_code == 0, capsys.readouterr().err
  # A fresh session per run was created and cleaned up...
  assert len(server.create_session_requests) == 2
  assert len(server.deleted_session_ids) == 2
  # ...and the empty id was never used as a session.
  assert all(req['sessionId'] for req in server.run_requests)
  assert '' not in server.deleted_session_ids


def _two_case_evalset_toml(
    *,
    user_a: str,
    session_a: str,
    user_b: str,
    session_b: str,
) -> str:
  """Builds a two-eval-case evalset whose cases pin the given sessions."""

  def case(eval_id: str, user_id: str, session_id: str, invocation_id: str) -> str:
    return f"""
[[eval_cases]]
eval_id = "{eval_id}"

[eval_cases.session_input]
app_name = "weather_agent"
user_id = "{user_id}"
session_id = "{session_id}"

[[eval_cases.conversation]]
invocation_id = "{invocation_id}"

[eval_cases.conversation.user_content]
role = "user"
parts = [ {{ text = "what is the weather in Tokyo?" }} ]

[eval_cases.conversation.final_response]
role = "model"
parts = [ {{ text = "It is sunny in Tokyo." }} ]

[eval_cases.conversation.intermediate_data]
tool_uses = [ {{ name = "get_weather", args = {{ city = "Tokyo" }} }} ]
"""

  return (
      'eval_set_id = "shared_set"\n'
      + case('case_a', user_a, session_a, 'inv-1')
      + case('case_b', user_b, session_b, 'inv-2')
  )


def test_two_cases_sharing_one_pinned_session_exits_two(
    tmp_path, capsys
) -> None:
  """Even with --num-runs 1, two cases on one session contaminate each other."""
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'shared.test.toml'
  evalset_path.write_text(
      _two_case_evalset_toml(
          user_a='cli_user',
          session_a='shared-session',
          user_b='cli_user',
          session_b='shared-session',
      ),
      encoding='utf-8',
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script() * 2
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

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'case_a' in err
  assert 'case_b' in err
  assert 'shared-session' in err
  # Rejected before anything was sent to the remote agent.
  assert server.run_requests == []
  assert not results_dir.exists()


def test_two_cases_pinning_different_sessions_are_allowed(
    tmp_path, capsys
) -> None:
  """Distinct pinned sessions do not share state, so they stay allowed."""
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'distinct.test.toml'
  evalset_path.write_text(
      _two_case_evalset_toml(
          user_a='cli_user',
          session_a='session-a',
          user_b='cli_user',
          session_b='session-b',
      ),
      encoding='utf-8',
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script() * 2
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
  used_sessions = {req['sessionId'] for req in server.run_requests}
  assert used_sessions == {'session-a', 'session-b'}


def test_same_session_id_under_different_users_is_allowed(
    tmp_path, capsys
) -> None:
  """A session id is only *the same* session when the user matches too."""
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'users.test.toml'
  evalset_path.write_text(
      # Same session id, different users -> two different remote sessions.
      _two_case_evalset_toml(
          user_a='alice',
          session_a='shared-session',
          user_b='bob',
          session_b='shared-session',
      ),
      encoding='utf-8',
  )
  server = FakeApiServer()
  # The fake server counts turns per session id, and both cases legitimately
  # use the id 'shared-session' (under different users), so script two turns
  # each rather than one.
  server.scripts['alice'] = _matching_script() * 2
  server.scripts['bob'] = _matching_script() * 2
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
  assert {req['userId'] for req in server.run_requests} == {'alice', 'bob'}


def test_duplicate_eval_id_within_evalset_exits_two(tmp_path, capsys) -> None:
  """add_eval_case() raises on a repeated eval_id; that must not be exit 1."""
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'dupe.test.toml'
  # Two cases, same eval_id -- schema-valid, but collides on registration.
  evalset_path.write_text(
      _WEATHER_EVAL_SET_TOML + _WEATHER_EVAL_SET_TOML.split('\n', 1)[1],
      encoding='utf-8',
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
  assert 'Duplicate eval_id' in err
  assert 'weather_case' in err
  assert 'weather_set' in err
  # A clean message, not a traceback.
  assert 'Traceback' not in err
  assert server.run_requests == []


def test_all_cases_failing_reports_no_results_saved(tmp_path, capsys) -> None:
  """With every case failing across several evalsets, no path is advertised.

  Complements the single-evalset 500 case: the saved-path line is driven by
  whether any eval set actually persisted results, so it must stay absent
  when several evalsets all fail.
  """
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  first = tmp_path / 'one.test.toml'
  first.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')
  second = tmp_path / 'two.test.toml'
  second.write_text(
      _WEATHER_EVAL_SET_TOML.replace('weather_set', 'weather_set_2'),
      encoding='utf-8',
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  server.fail_run_with_status = 500
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(first),
          str(second),
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
  captured = capsys.readouterr()
  assert 'Eval results saved under' not in captured.out
  assert 'No eval results were saved' in captured.err
  assert not results_dir.exists()


def test_partial_failure_still_reports_the_saved_path(tmp_path, capsys) -> None:
  """One evalset failing must not suppress the path the other one wrote.

  Guards the inverse mistake: the message is driven by "did anything save",
  not by "did anything fail", so a mixed run still points at the results.
  """
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  ok_set = tmp_path / 'ok.test.toml'
  ok_set.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')
  failing_set = tmp_path / 'failing.test.toml'
  failing_set.write_text(
      '''\
eval_set_id = "failing_set"

[[eval_cases]]
eval_id = "failing_case"

[eval_cases.session_input]
app_name = "weather_agent"
user_id = "broken_user"

[[eval_cases.conversation]]
invocation_id = "inv-1"

[eval_cases.conversation.user_content]
role = "user"
parts = [ { text = "what is the weather in Tokyo?" } ]

[eval_cases.conversation.final_response]
role = "model"
parts = [ { text = "It is sunny in Tokyo." } ]
''',
      encoding='utf-8',
  )

  server = FakeApiServer()
  # 'cli_user' is scripted and succeeds; 'broken_user' is set to fail
  # outright, so only the second evalset's case fails inference.
  server.scripts['cli_user'] = _matching_script()
  server.fail_user_ids = {'broken_user'}
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(ok_set),
          str(failing_set),
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

  captured = capsys.readouterr()
  # An inference failure happened, so exit 2 -- unchanged semantics.
  assert exit_code == 2
  assert 'INFERENCE FAILED' in captured.err
  # But the evalset that did succeed saved results, so the path is reported.
  assert 'Eval results saved under' in captured.out
  assert 'No eval results were saved' not in captured.err
  assert len(_saved_result_files(results_dir)) == 1


def test_repeated_header_values_all_reach_the_server(tmp_path) -> None:
  """A repeated header name must not collapse to its last value."""
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
          'Cookie: first=a',
          '--header',
          'Cookie: second=b',
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0
  assert server.received_header_pairs
  # Asserted on the raw pairs, not the dict: dict(request.headers) keeps only
  # the first value for a repeated name, which would hide the collapse this
  # test is about.
  for pairs in server.received_header_pairs:
    cookies = [value for name, value in pairs if name.lower() == 'cookie']
    assert cookies == ['first=a', 'second=b']


def test_drive_qualified_app_name_is_rejected(tmp_path, capsys) -> None:
  """'C:outside' has no separator but escapes --results-dir on Windows."""
  evalset_path = _write_evalset(tmp_path)
  server = FakeApiServer()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(evalset_path),
          '--app-name',
          'C:outside',
          '--results-dir',
          str(results_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  assert 'Refusing to use app name' in capsys.readouterr().err
  assert not results_dir.exists()
  assert server.run_requests == []


def test_non_string_session_id_is_rejected(tmp_path, capsys) -> None:
  """`session_id = 123` must never be silently downgraded to "unpinned".

  Which layer rejects it is environment-dependent: some google-adk builds
  validate extra-field types when the evalset is parsed, others let the int
  through to pytest-adk's own ``_pinned_session_id`` check (unit-tested
  directly in test_eval_service.py). Either is fine -- what must hold is that
  the run stops cleanly and never contacts the agent -- so this asserts that
  contract rather than the wording of one particular rejection.
  """
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'badtype.test.toml'
  evalset_path.write_text(
      _REUSED_SESSION_EVAL_SET_TOML.replace(
          'session_id = "pre-existing-session"', 'session_id = 123'
      ),
      encoding='utf-8',
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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  # A clean, actionable message either way -- never a traceback...
  assert 'Traceback' not in err
  assert 'session_id' in err or 'non-string' in err
  # ...and never silently run against a fresh session instead.
  assert server.run_requests == []
  assert not results_dir.exists()


def test_empty_string_session_id_still_means_unpinned(tmp_path, capsys) -> None:
  """The '' sentinel keeps working: only a wrong *type* is an error."""
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'empty.test.toml'
  evalset_path.write_text(
      _REUSED_SESSION_EVAL_SET_TOML.replace(
          'session_id = "pre-existing-session"', 'session_id = ""'
      ),
      encoding='utf-8',
  )
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
  assert len(server.create_session_requests) == 1


def test_eval_case_with_no_turns_exits_two(tmp_path, capsys) -> None:
  """`conversation = []` must not be scored as a successful inference."""
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'noturns.test.toml'
  evalset_path.write_text(
      'eval_set_id = "noturns_set"\n\n'
      '[[eval_cases]]\n'
      'eval_id = "noturns_case"\n'
      'conversation = []\n',
      encoding='utf-8',
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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  captured = capsys.readouterr()
  assert 'INFERENCE FAILED' in captured.err
  assert 'noturns_case' in captured.err
  # Never contacted the agent, and nothing was scored or saved.
  assert server.run_requests == []
  assert 'Eval results saved under' not in captured.out
  assert not results_dir.exists()


def test_pinned_session_with_state_is_rejected_by_preflight(
    tmp_path, capsys
) -> None:
  """The CLI rejects the combination before any inference runs."""
  _skip_without_session_id_support()
  (tmp_path / 'test_config.json').write_text(
      _TEST_CONFIG_JSON, encoding='utf-8'
  )
  evalset_path = tmp_path / 'conflict.test.toml'
  evalset_path.write_text(
      _REUSED_SESSION_EVAL_SET_TOML.replace(
          'session_id = "pre-existing-session"',
          'session_id = "pre-existing-session"\nstate = { locale = "en-US" }',
      ),
      encoding='utf-8',
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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'declares session_input.state' in err
  assert 'reuse_case' in err
  assert 'Traceback' not in err
  assert server.run_requests == []
  assert not results_dir.exists()


def test_user_id_with_slash_is_an_argparse_error(tmp_path, capsys) -> None:
  """A slashed --user-id can never be addressed, so fail before connecting."""
  evalset_path = _write_evalset(tmp_path)

  try:
    main([
        'eval',
        _AGENT_URL,
        str(evalset_path),
        '--user-id',
        'teams/acme',
    ])
    raised = False
  except SystemExit as e:
    raised = True
    assert e.code == 2

  assert raised
  assert '--user-id' in capsys.readouterr().err


def test_evalset_with_empty_criteria_exits_two(tmp_path, capsys) -> None:
  """Scoring against no metrics cannot fail, so it must not report success."""
  (tmp_path / 'test_config.json').write_text(
      json.dumps({'criteria': {}}), encoding='utf-8'
  )
  evalset_path = tmp_path / 'weather.test.toml'
  evalset_path.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')
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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'No eval metrics configured' in err
  assert 'weather_set' in err
  # Rejected before any inference, so no remote tool side effects were caused.
  assert server.run_requests == []
  assert not results_dir.exists()


def test_evalset_without_a_config_file_still_runs(tmp_path, capsys) -> None:
  """ADK's default criteria are non-empty, so a missing config is not empty.

  Guards the inverse mistake: the empty-criteria check must not reject
  evalsets that simply have no sibling test_config.json.
  """
  evalset_path = tmp_path / 'weather.test.toml'
  evalset_path.write_text(_WEATHER_EVAL_SET_TOML, encoding='utf-8')
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
          '--num-runs',
          '1',
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0, capsys.readouterr().err
  assert len(_saved_result_files(results_dir)) == 1


# --- Custom metrics (issue #11) ----------------------------------------------
#
# `get_eval_metrics_from_config()` puts a configured custom metric's function
# path on the EvalMetric, but nothing registers an evaluator for it. These
# exercise the registration the CLI now performs, and the up-front validation
# that keeps a bad config from being discovered only after the deployed agent
# has already run.


def _custom_metric_config_json(
    function_path: str, *, metric_name: str = 'quality', **custom_metric_extra
) -> str:
  """A test_config.json pairing one built-in metric with one custom metric."""
  return json.dumps({
      'criteria': {'tool_trajectory_avg_score': 1.0, metric_name: 0.5},
      'custom_metrics': {
          metric_name: {
              'code_config': {'name': function_path},
              **custom_metric_extra,
          }
      },
  })


def _run_custom_metric_eval(
    evalset_path: Path,
    results_dir: Path,
    server: FakeApiServer,
    *extra_args: str,
) -> int:
  return main(
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
          *extra_args,
      ],
      transport=_transport_for(server),
  )


def test_sync_custom_metric_scores_the_run_without_programmatic_setup(
    tmp_path, capsys, write_metric_module
) -> None:
  """The acceptance criterion: config alone is enough to run a custom metric."""
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json(f'{module_name}.sync_metric'),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = _run_custom_metric_eval(
      evalset_path,
      results_dir,
      server,
      '--pythonpath',
      str(metric_dir),
      '--print-detailed-results',
  )

  assert exit_code == 0, capsys.readouterr().err
  out = capsys.readouterr().out
  assert 'quality: score=0.75' in out
  # The built-in metric in the same criteria still uses its own evaluator.
  assert 'tool_trajectory_avg_score: score=1.0' in out
  assert len(_saved_result_files(results_dir)) == 1


def test_async_custom_metric_is_awaited_and_can_fail_the_run(
    tmp_path, capsys, write_metric_module
) -> None:
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json(f'{module_name}.async_metric'),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = _run_custom_metric_eval(
      evalset_path, results_dir, server, '--pythonpath', str(metric_dir)
  )

  # 0.25 against a 0.5 threshold: the metric ran, and failed on its own terms.
  assert exit_code == 1
  err = capsys.readouterr().err
  assert 'quality: score=0.25' in err
  # A metric failure is still a verdict, so the results are saved.
  assert len(_saved_result_files(results_dir)) == 1


# `write_metric_module`'s fixed source (tests/conftest.py) has no warning of
# its own, so this metric module is authored directly here rather than via
# that fixture -- everything else about it (a project-local module made
# importable via --pythonpath, one metric function) mirrors the fixture's
# module shape.
_NOISY_METRIC_MODULE_SOURCE = '''\
"""Project-local custom eval metric that also warns, like a real one might."""

import warnings

from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult

warnings.warn('metric says hi')


def noisy_metric(
    eval_metric, actual_invocations, expected_invocations, conversation_scenario
):
  return EvaluationResult(
      overall_score=0.75,
      overall_eval_status=EvalStatus.PASSED,
      per_invocation_results=[
          PerInvocationResult(
              actual_invocation=invocation, score=0.75, eval_status=EvalStatus.PASSED
          )
          for invocation in actual_invocations
      ],
  )
'''

_noisy_metric_module_counter = itertools.count()


def test_user_warning_from_a_custom_metric_still_surfaces(tmp_path, capsys, recwarn) -> None:
  """A UserWarning from project code is not caught by the google-adk filters.

  Uses `recwarn` (unlike `test_main_silences_adk_experimental_warnings`,
  which cannot): `recwarn` records *everything* regardless of category or
  message, so it is well-suited to asserting a warning is present, just not
  to asserting one has been suppressed.
  """
  metric_dir = tmp_path / 'lib'
  metric_dir.mkdir(parents=True, exist_ok=True)
  module_name = f'pytest_adk_noisy_metric_{next(_noisy_metric_module_counter)}'
  (metric_dir / f'{module_name}.py').write_text(
      _NOISY_METRIC_MODULE_SOURCE, encoding='utf-8'
  )
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json(f'{module_name}.noisy_metric'),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  try:
    exit_code = _run_custom_metric_eval(
        evalset_path, results_dir, server, '--pythonpath', str(metric_dir)
    )
  finally:
    sys.modules.pop(module_name, None)

  assert exit_code == 0, capsys.readouterr().err
  assert any(str(w.message) == 'metric says hi' for w in recwarn.list)


def test_custom_metric_module_is_found_in_the_working_directory(
    tmp_path, capsys, monkeypatch, write_metric_module
) -> None:
  """A console script does not put the invocation directory on sys.path.

  ``pytest-adk eval`` adds it, so a metric module sitting in the project the
  user runs the command from is importable without any extra flag.
  """
  project_dir = tmp_path / 'project'
  module_name = write_metric_module(project_dir)
  evalset_path = _write_evalset(
      project_dir / 'evals',
      _custom_metric_config_json(f'{module_name}.sync_metric'),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'
  monkeypatch.chdir(project_dir)

  exit_code = _run_custom_metric_eval(evalset_path, results_dir, server)

  assert exit_code == 0, capsys.readouterr().err


def test_sys_path_is_restored_after_the_command(
    tmp_path, monkeypatch, write_metric_module
) -> None:
  """main() is importable as a library function, so it must not leak sys.path."""
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json(f'{module_name}.sync_metric'),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  monkeypatch.chdir(tmp_path)
  before = list(sys.path)

  _run_custom_metric_eval(
      evalset_path,
      tmp_path / 'results',
      server,
      '--pythonpath',
      str(metric_dir),
  )

  assert sys.path == before


def test_configured_metric_info_reaches_the_registry(
    tmp_path, capsys, monkeypatch, write_metric_module
) -> None:
  """A custom score range outside [0, 1] must not be replaced by the default."""
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json(
          f'{module_name}.sync_metric',
          metric_info={
              'metric_name': 'quality',
              'description': 'Bounded at ten.',
              'metric_value_info': {
                  'interval': {'min_value': -10.0, 'max_value': 10.0}
              },
          },
      ),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()

  # The registry the CLI hands to RemoteEvalService is not otherwise
  # observable from the outside -- MetricInfo does not appear in the printed
  # results or the saved file -- so capture the object the command actually
  # scores with rather than inferring it from ADK's process-wide default.
  built_registries = []
  real_build = cli_module.build_metric_evaluator_registry

  def spy(*eval_configs):
    registry = real_build(*eval_configs)
    built_registries.append(registry)
    return registry

  monkeypatch.setattr(cli_module, 'build_metric_evaluator_registry', spy)

  exit_code = _run_custom_metric_eval(
      evalset_path,
      tmp_path / 'results',
      server,
      '--pythonpath',
      str(metric_dir),
      '--print-detailed-results',
  )

  assert exit_code == 0, capsys.readouterr().err
  assert 'quality: score=0.75' in capsys.readouterr().out
  assert len(built_registries) == 1
  registered = {
      metric_info.metric_name: metric_info
      for metric_info in built_registries[0].get_registered_metrics()
  }
  assert registered['quality'].description == 'Bounded at ten.'
  assert registered['quality'].metric_value_info.interval.max_value == 10.0


@pytest.mark.parametrize(
    'function_path_suffix, expected_fragment',
    [
        ('.does_not_exist', 'has no attribute'),
        ('.not_a_function', 'is not callable'),
    ],
)
def test_unresolvable_custom_metric_exits_two_before_inference(
    tmp_path,
    capsys,
    write_metric_module,
    function_path_suffix,
    expected_fragment,
) -> None:
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json(module_name + function_path_suffix),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = _run_custom_metric_eval(
      evalset_path, results_dir, server, '--pythonpath', str(metric_dir)
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'quality' in err
  assert expected_fragment in err
  # The point of validating up front: the deployed agent was never contacted,
  # so none of its tools had a side effect.
  assert server.run_requests == []
  assert not results_dir.exists()


def test_unimportable_custom_metric_module_exits_two_before_inference(
    tmp_path, capsys
) -> None:
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      _custom_metric_config_json('no_such_module_for_pytest_adk.metric'),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = _run_custom_metric_eval(evalset_path, results_dir, server)

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'no_such_module_for_pytest_adk' in err
  assert '--pythonpath' in err
  assert server.run_requests == []
  assert not results_dir.exists()


def test_criteria_metric_without_an_evaluator_exits_two_before_inference(
    tmp_path, capsys
) -> None:
  """A criteria name that is neither built-in nor declared as custom."""
  evalset_path = _write_evalset(
      tmp_path / 'evals',
      json.dumps({'criteria': {'mystery_metric': 0.5}}),
  )
  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = _run_custom_metric_eval(evalset_path, results_dir, server)

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'mystery_metric' in err
  assert 'weather_set' in err
  assert server.run_requests == []
  assert not results_dir.exists()


def test_two_evalsets_defining_one_metric_differently_exit_two(
    tmp_path, capsys, write_metric_module
) -> None:
  """One run shares one registry, so the two definitions cannot both apply."""
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)

  first_dir = tmp_path / 'first'
  _write_evalset(
      first_dir, _custom_metric_config_json(f'{module_name}.sync_metric')
  )
  second_dir = tmp_path / 'second'
  _write_evalset(
      second_dir, _custom_metric_config_json(f'{module_name}.async_metric')
  )
  # Distinct eval_set_ids, so this is rejected for the metric conflict rather
  # than by the duplicate-eval_set_id guard.
  (second_dir / 'weather.test.toml').write_text(
      _WEATHER_EVAL_SET_TOML.replace('weather_set', 'weather_set_two'),
      encoding='utf-8',
  )

  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(first_dir),
          str(second_dir),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
          '--num-runs',
          '1',
          '--pythonpath',
          str(metric_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 2
  err = capsys.readouterr().err
  assert 'quality' in err
  assert 'weather_set' in err and 'weather_set_two' in err
  assert server.run_requests == []


def test_two_evalsets_sharing_one_custom_metric_definition_run(
    tmp_path, capsys, write_metric_module
) -> None:
  """The inverse guard: identical definitions are not a conflict."""
  metric_dir = tmp_path / 'lib'
  module_name = write_metric_module(metric_dir)
  config_json = _custom_metric_config_json(f'{module_name}.sync_metric')

  first_dir = tmp_path / 'first'
  _write_evalset(first_dir, config_json)
  second_dir = tmp_path / 'second'
  _write_evalset(second_dir, config_json)
  (second_dir / 'weather.test.toml').write_text(
      _WEATHER_EVAL_SET_TOML.replace('weather_set', 'weather_set_two'),
      encoding='utf-8',
  )

  server = FakeApiServer()
  server.scripts['cli_user'] = _matching_script()
  results_dir = tmp_path / 'results'

  exit_code = main(
      [
          'eval',
          _AGENT_URL,
          str(first_dir),
          str(second_dir),
          '--app-name',
          _APP_NAME,
          '--user-id',
          'cli_user',
          '--results-dir',
          str(results_dir),
          '--num-runs',
          '1',
          '--pythonpath',
          str(metric_dir),
      ],
      transport=_transport_for(server),
  )

  assert exit_code == 0, capsys.readouterr().err
  assert len(_saved_result_files(results_dir)) == 2


def test_nonexistent_pythonpath_is_an_argparse_error(tmp_path, capsys) -> None:
  evalset_path = _write_evalset(tmp_path / 'evals')

  with pytest.raises(SystemExit) as excinfo:
    main([
        'eval',
        _AGENT_URL,
        str(evalset_path),
        '--app-name',
        _APP_NAME,
        '--pythonpath',
        str(tmp_path / 'missing'),
    ])

  assert excinfo.value.code == 2
  assert '--pythonpath' in capsys.readouterr().err
