# Copyright 2026 pytest-adk contributors
"""``pytest-adk`` command-line interface.

Currently exposes a single subcommand, ``eval``, which evaluates an ADK agent
running behind an ``adk api_server``-compatible HTTP endpoint. See
``REMOTE_EVAL_PLAN.md`` (sections 4, 5 "cli.py", 6 task R-3) for the design
behind this module.

Inference is delegated to the remote HTTP endpoint via
:class:`~pytest_adk.remote.client.AdkApiClient`; evalset loading (via
:func:`pytest_adk.evaluation._collect_eval_sets`), scoring, and result
persistence reuse the same local ADK evaluation machinery as the
``AgentEvaluator`` pytest fixture (:mod:`pytest_adk.evaluation`).

Import constraint: :mod:`pytest_adk.remote.eval_service` (``RemoteEvalService``)
is imported lazily, inside :func:`_run_eval`, rather than at module scope. On
google-adk v2 its import chain additionally requires ``vertexai`` (see that
module's docstring); importing it lazily means ``pytest-adk --help`` /
``pytest-adk eval --help`` keep working even where that dependency is
unavailable, and a missing-dependency error surfaces as a clean message on
stderr (exit code 2) instead of a traceback when ``eval`` is actually run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

import httpx
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_config import get_evaluation_criteria_or_default
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)

from .evaluation import _collect_eval_sets
from .remote.client import AdkApiClient

_DEFAULT_USER_ID = 'eval_user'
_DEFAULT_NUM_RUNS = 2
_DEFAULT_TIMEOUT = 300.0
_DEFAULT_PARALLELISM = 4
_DEFAULT_RESULTS_DIR = '.'
# Mirrors prompt_template._VALID_ENGINES / _DEFAULT_ENGINE. Spelled out here so
# argparse can offer `choices=` (and reject a bad value before any HTTP call);
# the pytest fixture gets the same choice from the
# `pytest_adk_prompt_template_engine` ini option instead, which this CLI
# deliberately does not read -- see the flag's help text.
_PROMPT_TEMPLATE_ENGINES = ('string', 'jinja')
_DEFAULT_PROMPT_TEMPLATE_ENGINE = 'string'

# Mirrors the private constant of the same name in evaluation.py: the subpath
# LocalEvalSetResultsManager writes ``*.evalset_result.json`` files into,
# relative to ``{results_dir}/{app_name}/``. Duplicated (rather than
# imported) because it is only used here to print the save location, not to
# derive behavior.
_ADK_EVAL_HISTORY_SUBDIR = Path('.adk') / 'eval_history'

EXIT_SUCCESS = 0
EXIT_METRIC_FAILURE = 1
EXIT_ERROR = 2


def _agent_url(value: str) -> str:
  """argparse ``type=`` for ``AGENT_URL``: a parseable http(s):// URL.

  The scheme prefix alone is not enough: values such as ``'http://['`` pass a
  prefix check but make ``httpx`` raise ``InvalidURL`` later, when the client
  is constructed. Parsing here turns those into an argparse error (which also
  exits with :data:`EXIT_ERROR`) instead of a traceback.
  """
  if not value.startswith(('http://', 'https://')):
    raise argparse.ArgumentTypeError(
        f"AGENT_URL must start with 'http://' or 'https://', got {value!r}."
    )
  try:
    httpx.URL(value)
  except httpx.InvalidURL as e:
    raise argparse.ArgumentTypeError(f'AGENT_URL is not a valid URL ({e}): {value!r}.')
  return value


def _positive_int(value: str) -> int:
  """argparse ``type=`` for an integer flag that must be >= 1."""
  try:
    parsed = int(value)
  except ValueError:
    raise argparse.ArgumentTypeError(f'must be an integer, got {value!r}.')
  if parsed < 1:
    raise argparse.ArgumentTypeError(f'must be >= 1, got {value!r}.')
  return parsed


def _header(value: str) -> tuple[str, str]:
  """argparse ``type=`` for a repeatable ``--header 'Name: Value'`` flag."""
  if ':' not in value:
    raise argparse.ArgumentTypeError(
        f"--header must be in 'Name: Value' form, got {value!r}."
    )
  name, _, header_value = value.partition(':')
  name = name.strip()
  header_value = header_value.strip()
  if not name:
    raise argparse.ArgumentTypeError(
        f"--header must have a non-empty header name, got {value!r}."
    )
  # httpx encodes header names/values as ASCII and raises UnicodeEncodeError
  # from the AsyncClient constructor otherwise. That construction happens
  # outside _run_eval's try block, so without this check a non-ASCII header
  # would abort with a traceback and exit 1 -- which automation reads as a
  # metric failure -- rather than the documented execution-error code.
  for part, label in ((name, 'name'), (header_value, 'value')):
    if not part.isascii():
      raise argparse.ArgumentTypeError(
          f'--header {label} must be ASCII (HTTP headers are not UTF-8),'
          f' got {value!r}.'
      )
  return name, header_value


def _build_parser() -> argparse.ArgumentParser:
  """Build the ``pytest-adk`` argument parser (``eval`` is the only subcommand)."""
  parser = argparse.ArgumentParser(
      prog='pytest-adk', description='pytest-adk command-line tools.'
  )
  subparsers = parser.add_subparsers(dest='command')

  eval_parser = subparsers.add_parser(
      'eval',
      help=(
          'Evaluate an ADK agent running behind an `adk api_server`-compatible'
          ' HTTP endpoint.'
      ),
      description=(
          'Runs each given evalset against a remote ADK agent: inference is'
          ' delegated to the HTTP endpoint (an `adk api_server`-compatible'
          ' REST API), while eval-data loading, scoring, and result'
          ' persistence reuse the same local ADK evaluation machinery as the'
          ' AgentEvaluator pytest fixture.'
      ),
  )
  eval_parser.add_argument(
      'agent_url',
      metavar='AGENT_URL',
      type=_agent_url,
      help="Base URL of the running api_server, e.g. 'http://localhost:8000'.",
  )
  eval_parser.add_argument(
      'eval_set_paths',
      metavar='EVAL_SET_PATH',
      nargs='+',
      help=(
          'Evalset file (.test.json/.test.toml) or directory, searched'
          ' recursively for such files (same convention as the'
          ' AgentEvaluator pytest fixture).'
      ),
  )
  eval_parser.add_argument(
      '--app-name',
      help=(
          'Name of the app on the remote server. If omitted, resolved via'
          ' GET /list-apps -- this only succeeds when exactly one app is'
          ' listed there.'
      ),
  )
  eval_parser.add_argument(
      '--config-file-path',
      help=(
          'Explicit EvalConfig (test_config.json), applied to every given'
          ' evalset. If omitted, each evalset uses the config discovered next'
          ' to it (same discovery convention as the AgentEvaluator pytest'
          ' fixture).'
      ),
  )
  eval_parser.add_argument(
      '--user-id',
      default=_DEFAULT_USER_ID,
      help=(
          'Default user_id used to create remote sessions (default:'
          f" {_DEFAULT_USER_ID!r}). An eval case's own session_input.user_id,"
          ' when set, takes precedence.'
      ),
  )
  eval_parser.add_argument(
      '--num-runs',
      type=_positive_int,
      default=_DEFAULT_NUM_RUNS,
      help=(
          f'Number of runs per eval case (default: {_DEFAULT_NUM_RUNS};'
          ' must be >= 1).'
      ),
  )
  eval_parser.add_argument(
      '--header',
      action='append',
      type=_header,
      default=[],
      dest='headers',
      metavar="'Name: Value'",
      help=(
          "Extra HTTP header sent with every request, e.g. 'Authorization:"
          " Bearer ...'. Repeatable."
      ),
  )
  eval_parser.add_argument(
      '--timeout',
      type=float,
      default=_DEFAULT_TIMEOUT,
      help=f'HTTP timeout in seconds (default: {_DEFAULT_TIMEOUT}).',
  )
  eval_parser.add_argument(
      '--parallelism',
      type=_positive_int,
      default=_DEFAULT_PARALLELISM,
      help=(
          'Inference/evaluation parallelism (default:'
          f' {_DEFAULT_PARALLELISM}; must be >= 1).'
      ),
  )
  eval_parser.add_argument(
      '--results-dir',
      default=_DEFAULT_RESULTS_DIR,
      help=(
          'Directory eval results are saved under, via'
          ' LocalEvalSetResultsManager (default: the current directory).'
          ' Results land in {results-dir}/{app-name}/.adk/eval_history/.'
      ),
  )
  eval_parser.add_argument(
      '--prompt-template-engine',
      choices=_PROMPT_TEMPLATE_ENGINES,
      default=_DEFAULT_PROMPT_TEMPLATE_ENGINE,
      help=(
          'Engine used to render <prompt:...> markers in the given evalsets'
          f' (default: {_DEFAULT_PROMPT_TEMPLATE_ENGINE!r}, i.e.'
          " string.Template's ${VAR}; 'jinja' selects Jinja2's {{ VAR }} and"
          ' needs the `jinja` extra). This is NOT read from the'
          ' pytest_adk_prompt_template_engine ini option -- the CLI does not'
          ' load pytest config -- so pass it explicitly to match that option'
          ' when the same evalsets are also run through the AgentEvaluator'
          ' fixture.'
      ),
  )
  eval_parser.add_argument(
      '--keep-sessions',
      action='store_true',
      help="Don't delete remote sessions created for this run afterwards.",
  )
  eval_parser.add_argument(
      '--print-detailed-results',
      action='store_true',
      help='Additionally print passing metric results, not just failures.',
  )
  return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
  """Entry point for the ``pytest-adk`` console script.

  Args:
      argv: Command-line arguments, excluding the program name. ``None`` uses
          ``sys.argv[1:]`` (argparse's default).
      transport: Private hook forwarded to ``AdkApiClient`` for the ``eval``
          subcommand. Tests inject ``httpx.MockTransport`` /
          ``httpx.ASGITransport`` here to run against a fake server; leave as
          ``None`` for a real HTTP connection.

  Returns:
      Process exit code: ``0`` if every eval metric passed, ``1`` if at least
      one metric failed, ``2`` on an execution error (bad ``AGENT_URL``,
      connection failure, ``--app-name`` resolution failure, evalset load
      failure, no evalset discovered at all, duplicate ``eval_set_id``s, or
      any eval case whose inference failed) or when no subcommand is given.
  """
  parser = _build_parser()
  args = parser.parse_args(argv)

  if args.command is None:
    parser.print_help()
    return EXIT_ERROR

  assert args.command == 'eval'  # the only subcommand there is
  return asyncio.run(_run_eval(args, transport=transport))


async def _run_eval(
    args: argparse.Namespace,
    *,
    transport: httpx.AsyncBaseTransport | None,
) -> int:
  """Implements the ``eval`` subcommand. See :func:`main` for the contract."""
  try:
    from .remote.eval_service import RemoteEvalService
  except ModuleNotFoundError as e:
    print(str(e), file=sys.stderr)
    return EXIT_ERROR

  headers = dict(args.headers) if args.headers else None
  try:
    client = AdkApiClient(
        args.agent_url,
        headers=headers,
        timeout=args.timeout,
        transport=transport,
    )
  except Exception as e:  # noqa: BLE001 - a bad client config is a user error
    # Defense in depth behind the argparse-level AGENT_URL/--header checks:
    # any other httpx rejection of the connection parameters should still be
    # an execution error rather than a traceback.
    print(f'Failed to set up the HTTP client: {e}', file=sys.stderr)
    return EXIT_ERROR

  try:
    app_name = await _resolve_app_name(client, args.app_name)
    if app_name is None:
      return EXIT_ERROR

    app_name_error = _app_name_path_error(app_name)
    if app_name_error is not None:
      print(app_name_error, file=sys.stderr)
      return EXIT_ERROR

    # The override is loaded *before* the evalsets and threaded into
    # _collect_eval_sets, which otherwise calls find_config_for_test_file()
    # for every file: an unloadable sibling test_config.json would abort the
    # run even though --config-file-path supplies the config meant to replace
    # it.
    override_config = None
    if args.config_file_path:
      try:
        override_config = get_evaluation_criteria_or_default(
            args.config_file_path
        )
      except Exception as e:  # noqa: BLE001 - surface any load failure to the user
        # A missing file, malformed JSON, or an unknown metric name in the
        # override config is a user-input error like an unloadable evalset,
        # and should exit EXIT_ERROR with a message rather than traceback.
        print(
            f'Failed to load --config-file-path {args.config_file_path!r}: {e}',
            file=sys.stderr,
        )
        return EXIT_ERROR

    eval_sets = _load_eval_sets(
        args.eval_set_paths,
        prompt_template_engine=args.prompt_template_engine,
        eval_config_override=override_config,
    )
    if eval_sets is None:
      return EXIT_ERROR

    empty_eval_set_ids = [
        eval_set.eval_set_id
        for eval_set, _ in eval_sets
        if not eval_set.eval_cases
    ]
    if empty_eval_set_ids:
      # perform_inference() yields nothing for a case-less eval set, so
      # success_results stays empty, evaluate() is skipped, and both failure
      # flags stay false -- the command would print that results were saved
      # and exit EXIT_SUCCESS having scored nothing.
      print(
          'Evalset(s) with no eval cases:'
          f' {", ".join(repr(i) for i in empty_eval_set_ids)}. There would be'
          ' nothing to run or score, so no result can be reported.',
          file=sys.stderr,
      )
      return EXIT_ERROR

    duplicate_eval_set_id = _find_duplicate_eval_set_id(eval_sets)
    if duplicate_eval_set_id is not None:
      # Registering two eval sets under one (app_name, eval_set_id) key makes
      # InMemoryEvalSetsManager.create_eval_set raise, and even if it did not,
      # inference requests identify an eval set by that id alone -- so the
      # second set would shadow the first and be scored with the wrong cases
      # and config. Reject it with an actionable message instead.
      print(
          f"Duplicate eval_set_id {duplicate_eval_set_id!r} across the given"
          ' EVAL_SET_PATHs. Eval set IDs must be unique within one run; give'
          ' each evalset file a distinct eval_set_id (or run them separately).',
          file=sys.stderr,
      )
      return EXIT_ERROR

    eval_sets_manager = InMemoryEvalSetsManager()
    for eval_set, _ in eval_sets:
      eval_sets_manager.create_eval_set(
          app_name=app_name, eval_set_id=eval_set.eval_set_id
      )
      for eval_case in eval_set.eval_cases:
        eval_sets_manager.add_eval_case(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            eval_case=eval_case,
        )

    # Deliberately NOT passed to RemoteEvalService/LocalEvalService: ADK's own
    # eval_set_results_manager-triggered auto-save inside evaluate() differs
    # by version (google-adk v1 saves once per InferenceResult; v2 saves once
    # per eval_set_id, batching all of them). With --num-runs > 1 that means a
    # version-dependent number of result files for the same eval_set. This
    # mirrors pytest_adk.evaluation._AgentEvaluator, which sidesteps the same
    # skew by collecting every run's EvalCaseResult itself and calling
    # save_eval_set_result() exactly once per eval_set.
    results_manager = LocalEvalSetResultsManager(agents_dir=args.results_dir)
    service = RemoteEvalService(
        client,
        app_name=app_name,
        eval_sets_manager=eval_sets_manager,
        default_user_id=args.user_id,
        keep_sessions=args.keep_sessions,
    )

    had_inference_failure = False
    had_metric_failure = False
    try:
      for eval_set, eval_config in eval_sets:
        set_had_inference_failure, set_had_metric_failure = (
            await _run_and_evaluate_eval_set(
                service,
                results_manager,
                app_name=app_name,
                eval_set=eval_set,
                eval_config=eval_config,
                num_runs=args.num_runs,
                parallelism=args.parallelism,
                print_detailed_results=args.print_detailed_results,
            )
        )
        had_inference_failure = (
            had_inference_failure or set_had_inference_failure
        )
        had_metric_failure = had_metric_failure or set_had_metric_failure
    except Exception as e:  # noqa: BLE001 - scoring/persistence failed outright
      # Scoring or writing the results failed (e.g. --results-dir is not
      # writable, so save_eval_set_result raises OSError). Without this, the
      # console script would exit 1 on the traceback -- indistinguishable to
      # automation from "a metric failed" -- even though no verdict was
      # actually reached.
      print(f'Evaluation failed: {e}', file=sys.stderr)
      return EXIT_ERROR
  finally:
    await client.aclose()

  eval_history_dir = Path(args.results_dir) / app_name / _ADK_EVAL_HISTORY_SUBDIR
  print(f'Eval results saved under: {eval_history_dir}')

  if had_inference_failure:
    return EXIT_ERROR
  if had_metric_failure:
    return EXIT_METRIC_FAILURE
  return EXIT_SUCCESS


async def _resolve_app_name(
    client: AdkApiClient, app_name: str | None
) -> str | None:
  """Returns ``app_name`` as-is, or resolves it via ``GET /list-apps``.

  Args:
      client: Client used to query the remote server.
      app_name: The ``--app-name`` value, or ``None`` if omitted.

  Returns:
      The app name to use, or ``None`` if it could not be resolved (an error
      has already been printed to stderr in that case).
  """
  if app_name:
    return app_name

  try:
    apps = await client.list_apps()
  except (httpx.HTTPError, ValueError) as e:
    # ValueError covers a body that is not JSON at all (json.JSONDecodeError)
    # and one whose shape is not a list of app names (raised by list_apps) --
    # e.g. AGENT_URL pointing at a non-api_server, or a redirect to an HTML
    # page. Both are --app-name resolution failures, not tracebacks.
    print(f'Failed to resolve --app-name via GET /list-apps: {e}', file=sys.stderr)
    return None

  if len(apps) == 1:
    return apps[0]

  if not apps:
    print(
        'Could not resolve --app-name automatically: the remote server'
        ' reports no apps (GET /list-apps returned an empty list). Pass'
        ' --app-name explicitly.',
        file=sys.stderr,
    )
  else:
    print(
        'Could not resolve --app-name automatically: the remote server'
        f' reports {len(apps)} apps. Pass one of these explicitly via'
        f' --app-name: {", ".join(sorted(apps))}',
        file=sys.stderr,
    )
  return None


def _app_name_path_error(app_name: str) -> str | None:
  """Returns an error message if ``app_name`` is unsafe as a path segment.

  ``LocalEvalSetResultsManager`` builds ``{results_dir}/{app_name}/.adk/
  eval_history/`` from this value, so a name containing a separator or a
  traversal segment would write results outside ``--results-dir``. That
  matters because the name can come from the *remote* server's
  ``GET /list-apps``, not just from ``--app-name``.

  google-adk v2 rejects such names itself (``validate_path_segment``), but
  v1 does not -- it happily writes outside the directory -- so this check is
  what makes the behavior the same on both. Its rules mirror v2's.

  Returns:
      ``None`` if the name is safe, otherwise a ready-to-print message.
  """
  reason = None
  if not app_name:
    reason = 'must not be empty'
  elif '\x00' in app_name:
    reason = 'must not contain null bytes'
  elif '/' in app_name or '\\' in app_name:
    reason = 'must not contain path separators'
  elif app_name in ('.', '..'):
    reason = 'must not be a path traversal segment'
  if reason is None:
    return None
  return (
      f'Refusing to use app name {app_name!r}: it {reason}, and eval results'
      ' are written to {results-dir}/{app-name}/.adk/eval_history/. Pass a'
      ' plain app name via --app-name.'
  )


def _load_eval_sets(
    eval_set_paths: Sequence[str],
    *,
    prompt_template_engine: str = _DEFAULT_PROMPT_TEMPLATE_ENGINE,
    eval_config_override: EvalConfig | None = None,
) -> list[tuple[EvalSet, EvalConfig]] | None:
  """Loads every evalset from ``eval_set_paths`` via ``_collect_eval_sets``.

  Args:
      eval_set_paths: ``EVAL_SET_PATH`` positional arguments, each an evalset
          file or a directory searched recursively.
      prompt_template_engine: Engine used to render ``<prompt:...>`` markers,
          from ``--prompt-template-engine``.
      eval_config_override: ``--config-file-path``'s already-loaded config,
          which replaces sibling ``test_config.json`` discovery when set.

  Returns:
      The concatenated ``(EvalSet, EvalConfig)`` pairs, in argument order, or
      ``None`` if any path failed to load (an error has already been printed
      to stderr in that case).
  """
  eval_sets: list[tuple[EvalSet, EvalConfig]] = []
  for eval_set_path in eval_set_paths:
    try:
      path_eval_sets = _collect_eval_sets(
          eval_set_path,
          prompt_template_engine=prompt_template_engine,
          eval_config_override=eval_config_override,
      )
    except Exception as e:  # noqa: BLE001 - surface any load failure to the user
      print(f'Failed to load evalset(s) from {eval_set_path!r}: {e}', file=sys.stderr)
      return None

    # Checked per path, not just on the aggregate: with several EVAL_SET_PATHs
    # a populated one would otherwise mask a directory that contributed
    # nothing (e.g. its files are misnamed), silently evaluating a subset and
    # still exiting EXIT_SUCCESS.
    if not path_eval_sets:
      print(
          f'No evalsets found in {eval_set_path!r}. Directories are searched'
          ' recursively for files named *.test.json / *.test.toml.',
          file=sys.stderr,
      )
      return None

    eval_sets.extend(path_eval_sets)
  return eval_sets


def _find_duplicate_eval_set_id(
    eval_sets: Sequence[tuple[EvalSet, EvalConfig]],
) -> str | None:
  """Returns the first ``eval_set_id`` that occurs more than once, else ``None``."""
  seen: set[str] = set()
  for eval_set, _ in eval_sets:
    if eval_set.eval_set_id in seen:
      return eval_set.eval_set_id
    seen.add(eval_set.eval_set_id)
  return None


async def _run_and_evaluate_eval_set(
    service,
    results_manager: LocalEvalSetResultsManager,
    *,
    app_name: str,
    eval_set: EvalSet,
    eval_config: EvalConfig,
    num_runs: int,
    parallelism: int,
    print_detailed_results: bool,
) -> tuple[bool, bool]:
  """Runs inference + evaluation for one eval set, saves, and reports results.

  Runs ``num_runs`` ``InferenceRequest``s (mirroring
  ``AgentEvaluator._get_eval_results_by_eval_id``'s "repeat the request
  num_runs times" approach). Inference ``FAILURE`` results are reported
  directly (eval case id + error message, to stderr) and excluded from
  ``evaluate()`` -- passing a ``FAILURE`` result (``inferences=None``) into
  ``evaluate()`` is known to raise ``TypeError`` on pre-v2 google-adk (see
  ``tests/remote/test_eval_service.py``'s
  ``_evaluate_tolerating_known_adk_v1_none_inferences_bug`` for background);
  excluding them sidesteps that entirely while also giving cleaner per-case
  reporting than a mixed-in eval failure would.

  All ``EvalCaseResult``s (across every run) are saved via a single
  ``results_manager.save_eval_set_result()`` call after ``evaluate()``
  finishes, rather than letting ``RemoteEvalService``/``LocalEvalService``
  auto-save internally -- see the comment where this function is called for
  why.

  Args:
      service: The ``RemoteEvalService`` to run inference/evaluation with.
      results_manager: Where to save this eval set's results.
      app_name: The (local ``eval_sets_manager``-namespaced) app name.
      eval_set: The eval set to run.
      eval_config: The ``EvalConfig`` (metrics/thresholds) for this eval set.
      num_runs: Number of ``InferenceRequest``s to issue for this eval set.
      parallelism: Value forwarded to ``InferenceConfig``/``EvaluateConfig``.
      print_detailed_results: Whether to also print passing metric results.

  Returns:
      A ``(had_inference_failure, had_metric_failure)`` tuple for this eval
      set.
  """
  inference_results: list[InferenceResult] = []
  for _ in range(num_runs):
    inference_request = InferenceRequest(
        app_name=app_name,
        eval_set_id=eval_set.eval_set_id,
        inference_config=InferenceConfig(parallelism=parallelism),
    )
    async for inference_result in service.perform_inference(inference_request):
      inference_results.append(inference_result)

  had_inference_failure = False
  success_results: list[InferenceResult] = []
  for inference_result in inference_results:
    if inference_result.status == InferenceStatus.SUCCESS:
      success_results.append(inference_result)
    else:
      had_inference_failure = True
      print(
          f"INFERENCE FAILED for eval set '{eval_set.eval_set_id}' eval case"
          f" '{inference_result.eval_case_id}': {inference_result.error_message}",
          file=sys.stderr,
      )

  had_metric_failure = False
  if success_results:
    eval_metrics = get_eval_metrics_from_config(eval_config)
    evaluate_request = EvaluateRequest(
        inference_results=success_results,
        evaluate_config=EvaluateConfig(
            eval_metrics=eval_metrics, parallelism=parallelism
        ),
    )
    eval_case_results = [
        eval_case_result
        async for eval_case_result in service.evaluate(evaluate_request)
    ]
    results_manager.save_eval_set_result(
        app_name=app_name,
        eval_set_id=eval_set.eval_set_id,
        eval_case_results=eval_case_results,
    )
    for eval_case_result in eval_case_results:
      if _print_eval_case_result(
          eval_case_result, print_detailed_results=print_detailed_results
      ):
        had_metric_failure = True

  return had_inference_failure, had_metric_failure


def _print_eval_case_result(
    eval_case_result: EvalCaseResult, *, print_detailed_results: bool
) -> bool:
  """Prints per-metric results for one eval case.

  Failing metrics are always printed (to stderr); passing metrics are only
  printed when ``print_detailed_results`` is set (to stdout).

  Args:
      eval_case_result: The scored eval case to report.
      print_detailed_results: Whether to also print passing metric results.

  Returns:
      ``True`` if this eval case did not pass overall
      (``final_eval_status != EvalStatus.PASSED``).
  """
  case_failed = eval_case_result.final_eval_status != EvalStatus.PASSED
  for metric_result in eval_case_result.overall_eval_metric_results:
    metric_passed = metric_result.eval_status == EvalStatus.PASSED
    if metric_passed and not print_detailed_results:
      continue
    status_label = 'PASSED' if metric_passed else metric_result.eval_status.name
    print(
        f'[{eval_case_result.eval_id}] {metric_result.metric_name}:'
        f' score={metric_result.score} threshold={metric_result.threshold}'
        f' status={status_label}',
        file=sys.stdout if metric_passed else sys.stderr,
    )
  return case_failed


if __name__ == '__main__':
  sys.exit(main())
