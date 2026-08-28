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
module's docstring), which pytest-adk declares as a normal dependency itself;
importing it lazily means ``pytest-adk --help`` / ``pytest-adk eval --help``
keep working even in an environment where that dependency is somehow missing,
and a missing-dependency error surfaces as a clean message on stderr (exit
code 2) instead of a traceback when ``eval`` is actually run.

Output: google-adk's own deprecation and ``[EXPERIMENTAL]`` ``UserWarning``s
are suppressed by default (see :func:`_silence_google_adk_warnings`, which
also documents the one warning that escapes this on google-adk 1.30.0 and
1.31.0, because it fires before ``main()`` runs at all); set
``PYTHONWARNINGS=always::UserWarning`` in the environment to see them again.
Note that the interpreter's ``-W`` option is *not* usable for this when
running the ``pytest-adk`` console script: ``-W`` has to be consumed by
``python`` itself, and ``pytest-adk eval -W ...`` would just hand ``-W`` to
this module's argparse parser, which rejects it. ``PYTHONWARNINGS`` is the
supported opt-out.

Warnings raised by your own agent or custom metric code are unaffected, with
one deliberate exception: the ``[EXPERIMENTAL]`` rule matches on message text
alone (it has to -- see :func:`_silence_google_adk_warnings`), so a
``UserWarning`` of your own whose message *starts with* ``[EXPERIMENTAL]``
is suppressed too.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import ntpath
import os
import sys
import warnings
from pathlib import Path
from typing import Iterator
from typing import NamedTuple
from typing import Sequence

import httpx
from google.adk.evaluation import AgentEvaluator as _AdkAgentEvaluator
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_config import get_evaluation_criteria_or_default
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)

from .evaluation import _collect_eval_sets
from .evaluation import _ReadableNameEvalSetResultsManager
from .metrics import build_metric_evaluator_registry
from .metrics import check_criteria_have_evaluators
from .metrics import check_custom_metrics_are_consistent
from .remote.client import AdkApiClient


def _active_warning_filters() -> list:
  """The filter list that ``warnings.filterwarnings`` actually mutates.

  Usually this is the module-global ``warnings.filters``. Python 3.14 added
  context-aware warnings (``sys.flags.context_aware_warnings``, on by default
  in free-threaded builds), where an active ``catch_warnings()`` moves the
  filters into a context variable so the block is context- and thread-safe;
  the module global then goes stale, and appending to or removing from it has
  no effect on what ``warnings.warn`` consults. ``warnings._get_filters()``
  returns whichever list is live. It is private, but it is the only accessor
  there is, and the ``getattr`` fallback covers every Python before 3.14,
  where the global is the only list that exists.
  """
  get_filters = getattr(warnings, '_get_filters', None)
  return get_filters() if get_filters is not None else warnings.filters


def _silence_google_adk_warnings() -> tuple[tuple, ...]:
  """Suppresses google-adk-originated ``UserWarning``s for CLI output.

  google-adk's evaluation import chain is noisy by default: importing
  ``pytest_adk.remote.eval_service`` (see :func:`_run_eval`) drags in a
  ``vertexai.preview.rag`` deprecation warning with no distinguishing prefix,
  plus several ``[EXPERIMENTAL] ...`` notices from ADK's ``@experimental``
  decorator. None of this is actionable by a ``pytest-adk eval`` user, so it
  is filtered out here rather than left to print on every run.

  Two separate filters are needed, not one:

  - The ``module=`` filter alone would miss the ``[EXPERIMENTAL]`` warnings
    raised by pytest-adk's own call into ``LocalEvalService.__init__``:
    ``@experimental`` warns with ``stacklevel=2``, so Python attributes that
    warning to its *caller*'s module -- ``pytest_adk.remote.eval_service`` --
    not to ``google.adk``.
  - The ``message=`` filter alone would miss the ``vertexai.preview.rag``
    deprecation warning, which carries no ``[EXPERIMENTAL]`` prefix.

  The ``module`` pattern is deliberately ``google.adk.dependencies`` and not
  ``google.adk`` at large. A ``module=`` filter is matched against the module
  the warning is *attributed* to, which ``stacklevel`` moves up the stack --
  so a broad ``google\\.adk`` pattern also covers
  ``google.adk.evaluation.custom_metric_evaluator``, which is exactly where
  ADK calls a user's custom metric function from. A metric that warned with
  ``stacklevel=2`` (idiomatic: point at the caller, not at the metric's own
  line) would then be silently swallowed even though its message has no
  ``[EXPERIMENTAL]`` prefix. Narrowing to ``google.adk.dependencies`` keeps
  that from happening: no user code is ever called from there, so nothing
  but ADK's own dependency shims can be attributed to it.

  Checked against every google-adk in the supported ``>=1.30.0,<3`` range
  (1.30.0, 1.31.0, 1.33.0, 1.35.0, 1.37.0, 2.0.0, 2.5.0):
  ``google.adk.dependencies.vertexai`` is the only ``google.adk`` module that
  emits a ``UserWarning`` without an ``[EXPERIMENTAL]`` prefix, so this
  narrower pattern suppresses everything the broad one usefully did. The
  ``(\\.|$)`` guard matches that package and its submodules but not an
  unrelated one that merely starts with the same characters.

  Deliberately scoped to ``category=UserWarning`` only, and not keyed on
  ``pytest_adk.remote.eval_service`` as a module: a genuine ``UserWarning``
  raised by the user's own agent code or a custom metric module (which can
  live in that same call stack) must stay visible.

  Accepted cost of the message-only rule: because it cannot also key on a
  module (per the first bullet above), it matches purely on message text,
  so a ``UserWarning`` raised by the user's own code whose message begins
  with ``[EXPERIMENTAL]`` -- e.g. a custom metric announcing an experimental
  API of its own -- is suppressed as well. This is judged an acceptable
  trade against dropping ADK's ``[EXPERIMENTAL]`` suppression entirely, and
  is called out in this module's docstring and in the README so the
  narrowed guarantee is the documented one.

  ``append=True`` on both filters places them after Python's five default
  filters (none of which match ``UserWarning``) and after any filter the
  user supplied via ``-W`` / ``PYTHONWARNINGS`` (those are installed at
  interpreter startup, so they are already in ``warnings.filters`` by the
  time ``main()`` runs), so e.g. ``PYTHONWARNINGS=always::UserWarning``
  still wins over these. That environment variable -- not ``-W``, which
  ``python`` would have to consume and which the ``pytest-adk`` console
  script therefore cannot accept -- is the documented opt-out; see this
  module's docstring.

  google-adk also offers an ``ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS``
  environment variable, considered and rejected here: it does not cover the
  vertexai deprecation warning, it mutates a process-global environment
  variable the caller may itself depend on, and -- unlike a ``warnings``
  filter -- it cannot be overridden by ``-W``.

  Known accepted limitation, on google-adk 1.30.0 and 1.31.0 only (1.32.0
  stopped emitting it; those two are the affected releases in the supported
  ``>=1.30.0,<3`` range -- there is no 1.30.1): one
  ``[EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH is enabled.`` warning
  is emitted from ``google/adk/features/_feature_decorator.py`` while
  ``google.adk`` is being imported -- which happens during ``import
  pytest_adk`` itself, before this function or even ``main()`` can run. It is
  therefore not reachable from here, and suppressing it would mean installing
  a filter at ``pytest_adk`` import time, which would silently apply to the
  pytest plugin path and to every library consumer of the package rather than
  just this CLI. Left as-is and documented in the README instead; upgrading
  google-adk is the actual fix.

  Returns:
      The filter entries actually appended to ``warnings.filters``, in the
      order they were added, so :func:`_google_adk_warnings_silenced` can
      take exactly those back out again. The list length is checked rather
      than assuming each call appended: ``filterwarnings(append=True)`` is a
      no-op when an identical entry is already present, and treating
      ``filters[-1]`` as ours regardless would hand the caller somebody
      else's filter to delete.
  """
  installed = []
  for keyed_on in (
      {'message': r'\[EXPERIMENTAL\]'},
      {'module': r'google\.adk\.dependencies(\.|$)'},
  ):
    active = _active_warning_filters()
    before = len(active)
    warnings.filterwarnings(
        'ignore', category=UserWarning, append=True, **keyed_on
    )
    if len(active) > before:
      installed.append(active[-1])
  return tuple(installed)


@contextlib.contextmanager
def _google_adk_warnings_silenced() -> Iterator[None]:
  """Applies :func:`_silence_google_adk_warnings` for the duration of a block.

  Removes exactly the two filters it added, rather than wrapping the block in
  ``warnings.catch_warnings()``. Both approaches stop the filters leaking out
  of an in-process ``main()`` call -- which matters because
  ``filterwarnings(message=...)`` entries carry a freshly compiled regex and
  so never compare equal to one another, meaning repeated calls would
  otherwise grow ``warnings.filters`` without bound -- but ``catch_warnings()``
  restores a *snapshot*, throwing away every unrelated change made in between.
  A custom metric module is imported inside this block (see ``--pythonpath``),
  and a module-level ``warnings.filterwarnings(...)`` in it is a normal thing
  for library code to do; with a snapshot restore that registration is
  discarded, and because the module stays in ``sys.modules`` its body never
  re-runs to reinstate it. Same for anything else the block imports, and for
  ``warnings.showwarning``, which ``catch_warnings()`` also rolls back.

  Removal goes through :func:`_active_warning_filters` for the same reason
  the install does: on context-aware-warnings builds the live list may not be
  ``warnings.filters``, and removing from the stale global would silently do
  nothing, leaving the filters in place for the rest of the process.

  Entries are matched by *identity*, not equality, and ``list.remove()`` is
  therefore not usable. Filter entries are plain tuples, so two of them
  compare equal whenever their fields do -- and the compiled-pattern field
  does not save us, because ``re.compile`` memoises: compiling the same
  pattern with the same flags twice hands back the very same object. A
  custom metric that registers, say,
  ``filterwarnings('ignore', message=r'\\[EXPERIMENTAL\\]',
  category=UserWarning)`` therefore builds a tuple equal to the one this
  block installed. ``filterwarnings`` with the default ``append=False``
  *removes any equal entry* before inserting at the front, so at that point
  the entry here is already gone from the list and the metric's replacement
  stands in its place -- and an equality-based removal would delete that
  replacement, losing a registration the metric cannot make again (its
  module stays in ``sys.modules``, so its body never re-runs). Identity
  leaves the replacement alone and removes nothing, which is right: what
  this block installed is no longer there.

  A missing entry is tolerated for the same reason, and because the block
  may legitimately have called ``warnings.resetwarnings()``.

  ``_filters_mutated()`` is private but is what the stdlib's own
  ``catch_warnings`` calls on exit, and is present on every supported Python
  (3.10-3.14). It bumps the filter-version counter that invalidates each
  module's ``__warningregistry__``; without it, a warning that these filters
  suppressed during the block could stay cached as "already handled" and
  remain invisible after they are gone.
  """
  installed = _silence_google_adk_warnings()
  try:
    yield
  finally:
    active = _active_warning_filters()
    for entry in installed:
      for index, present in enumerate(active):
        if present is entry:
          del active[index]
          break
    warnings._filters_mutated()


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


class _EvalSetOutcome(NamedTuple):
  """What running and scoring one eval set produced.

  ``saved_results`` exists because the other two flags cannot distinguish
  "everything passed" from "nothing was scorable": when every inference fails,
  there is nothing to hand to ``evaluate()``, so no result file is written and
  both failure flags stay false. The caller needs that difference to avoid
  pointing at a results path that was never created.
  """

  had_inference_failure: bool
  had_metric_failure: bool
  saved_results: bool


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


def _user_id(value: str) -> str:
  """argparse ``type=`` for ``--user-id``: usable as one REST path segment.

  The api_server routes ``/users/{user_id}/`` as a single path segment and an
  encoded slash is decoded back into a separator before routing, so a slashed
  user id cannot be addressed at all. ``AdkApiClient`` rejects it too (which
  is what protects a per-eval-case ``session_input.user_id``); catching the
  global flag here turns it into an argparse error before any HTTP call
  instead of an identical failure on every eval case.
  """
  if '/' in value:
    raise argparse.ArgumentTypeError(
        f'--user-id must not contain "/", got {value!r}.'
    )
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


def _importable_dir(value: str) -> str:
  """argparse ``type=`` for ``--pythonpath``: an existing directory.

  A path that does not exist would be added to ``sys.path`` without effect,
  and the resulting failure ("could not import module ...") would point at the
  metric rather than at the typo that caused it.
  """
  if not os.path.isdir(value):
    raise argparse.ArgumentTypeError(
        f'--pythonpath must be an existing directory, got {value!r}.'
    )
  return os.path.abspath(value)


@contextlib.contextmanager
def _importable_from(extra_paths: Sequence[str]) -> Iterator[None]:
  """Makes ``extra_paths`` and the working directory importable, temporarily.

  Custom metric functions are named by import path (``code_config.name``), and
  they usually live in the project being evaluated. A console script -- unlike
  ``python -m`` or ``python script.py`` -- does *not* put the invocation
  directory on ``sys.path``, so without this a project-local metric module is
  simply not importable and the run fails on a config that looks correct.

  ``--pythonpath`` entries come first, then the working directory, so an
  explicitly pointed-at directory wins over an accidental same-named module
  next to the shell's cwd.

  ``sys.path`` is restored on exit: ``main()`` is called in-process by the test
  suite (and is importable as a library function), so it should not permanently
  reshape the caller's import resolution. Modules already imported stay in
  ``sys.modules``, which is what makes the later, in-scoring lookup by
  ``_CustomMetricEvaluator`` resolve to the same function object.
  """
  original_sys_path = list(sys.path)
  for path in reversed([*extra_paths, os.getcwd()]):
    sys.path.insert(0, path)
  try:
    yield
  finally:
    sys.path[:] = original_sys_path


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
      type=_user_id,
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
      '--pythonpath',
      action='append',
      type=_importable_dir,
      default=[],
      dest='pythonpath',
      metavar='PATH',
      help=(
          'Extra directory to import custom metric functions from'
          ' (`custom_metrics` in the eval config). Repeatable. The directory'
          ' the command runs in is always importable; use this for metric'
          ' modules that live elsewhere.'
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
      help=(
          'Report more than just the one-line failures. Adds (a) a one-line'
          ' result for each *passing* metric, on stdout, and (b) a'
          ' per-invocation breakdown table for each metric that did not pass,'
          ' on stderr, showing the prompt alongside the expected and actual'
          ' response and tool calls. Note this is not the same setting as'
          " google-adk's own `print_detailed_results` (which the"
          ' `AgentEvaluator` pytest fixture takes): that one defaults to true'
          ' and only prints the failure tables.'
      ),
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
      failure, no evalset discovered at all, duplicate ``eval_set_id``s, a
      custom metric that cannot be resolved, a criteria metric with no
      evaluator, or any eval case whose inference failed) or when no
      subcommand is given.
  """
  # Takes the two silencing filters back out on the way out, so they do not
  # stick around across in-process main() calls (the test suite makes ~40 of
  # them, and each would otherwise grow `warnings.filters` by two entries --
  # see _google_adk_warnings_silenced, which also explains why this is not
  # `warnings.catch_warnings()`). This covers the `--help` / no-subcommand
  # paths below, which return early via SystemExit-free `return` statements
  # -- and argparse's own `--help` SystemExit still unwinds the context
  # manager normally, cleaning up either way.
  with _google_adk_warnings_silenced():
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
      parser.print_help()
      return EXIT_ERROR

    assert args.command == 'eval'  # the only subcommand there is
    # Wrapped here rather than inside _run_eval so the eval config's custom
    # metric modules are importable for the whole subcommand, including the
    # scoring that happens after inference.
    with _importable_from(args.pythonpath):
      return asyncio.run(_run_eval(args, transport=transport))


async def _run_eval(
    args: argparse.Namespace,
    *,
    transport: httpx.AsyncBaseTransport | None,
) -> int:
  """Implements the ``eval`` subcommand. See :func:`main` for the contract."""
  try:
    # _pinned_session_id comes from the same module (and the same lazy import,
    # for the vertexai-dependency reason documented in this module's
    # docstring) as the runtime that acts on it, so the guard below and the
    # session-creation branch in _perform_remote_inference_single_eval_item
    # share one predicate by construction and cannot drift apart.
    from .remote.eval_service import _group_eval_cases_by_pinned_session
    from .remote.eval_service import _pinned_session_id
    from .remote.eval_service import _pinned_session_state_error
    from .remote.eval_service import RemoteEvalService
  except ModuleNotFoundError as e:
    print(str(e), file=sys.stderr)
    return EXIT_ERROR

  # A list of pairs, not a dict: HTTP allows a header name to repeat (two
  # Cookie fields, a repeated proxy scope header), --header is documented as
  # repeatable, and httpx accepts the pair sequence as-is. dict() would keep
  # only the last occurrence and silently alter auth/routing context.
  headers = list(args.headers) if args.headers else None
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

    # Enforced here rather than in RemoteEvalService because the service never
    # sees num_runs: the CLI drives the repeat loop itself (see
    # _run_and_evaluate_eval_set), so this is the only layer that can observe
    # the conflict.
    # Run the shared predicate over every eval case once, up front: it rejects
    # a non-string session_id (google-adk v2 keeps extra fields without
    # validating their types), and both guards below call it, so validating
    # here keeps that ValueError from escaping either of them as a traceback.
    try:
      for eval_set, _ in eval_sets:
        for eval_case in eval_set.eval_cases:
          pinned = _pinned_session_id(eval_case)
          # A pinned session is never created, so a declared initial state
          # would silently never be applied. Same shared-helper treatment as
          # the predicate itself, so preflight and runtime agree.
          state_error = _pinned_session_state_error(eval_case, pinned)
          if state_error is not None:
            raise ValueError(state_error)
    except ValueError as e:
      print(str(e), file=sys.stderr)
      return EXIT_ERROR

    if args.num_runs > 1:
      reused_session_case_ids = [
          eval_case.eval_id
          for eval_set, _ in eval_sets
          for eval_case in eval_set.eval_cases
          if _pinned_session_id(eval_case) is not None
      ]
      if reused_session_case_ids:
        print(
            '--num-runs'
            f' {args.num_runs} cannot be combined with eval cases that reuse'
            ' an existing remote session via session_input.session_id:'
            f' {", ".join(repr(i) for i in reused_session_case_ids)}. Every'
            ' run would send the conversation to that same mutable session,'
            " so later runs would see earlier runs' turns, state changes and"
            ' tool side effects instead of being independent repetitions.'
            ' Pass --num-runs 1, or drop session_input.session_id to let each'
            ' run get a fresh session.',
            file=sys.stderr,
        )
        return EXIT_ERROR

    # Checked regardless of --num-runs, and across every loaded evalset: one
    # RemoteEvalService drives them all, so two eval cases resolving to the
    # same (user_id, session_id) share one mutable remote session whether
    # perform_inference() runs them concurrently or a later evalset reaches
    # the session sequentially. Either way each case sees the other's turns,
    # state changes and tool side effects.
    shared_session = _eval_cases_sharing_a_pinned_session(
        eval_sets,
        default_user_id=args.user_id,
        group_by_pinned_session=_group_eval_cases_by_pinned_session,
    )
    if shared_session is not None:
      (user_id, session_id), case_ids = shared_session
      print(
          'Eval cases'
          f' {", ".join(repr(i) for i in case_ids)} all pin the same remote'
          f' session (user_id {user_id!r}, session_input.session_id'
          f' {session_id!r}). They would share one mutable session and'
          " contaminate each other's turns, state changes and tool side"
          ' effects. Give each eval case its own session_input.session_id,'
          ' drop it so each gets a fresh session, or run them separately.',
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

    for eval_set, eval_config in eval_sets:
      if not get_eval_metrics_from_config(eval_config):
        # A schema-valid config whose `criteria` is empty. Left alone, the run
        # would perform real inference (with real remote tool side effects),
        # then score against no metrics at all -- which cannot fail -- save a
        # result file and exit 0, reporting success for something that was
        # never checked. Note this cannot fire for an evalset with no config
        # file: ADK's default criteria are non-empty.
        print(
            f"No eval metrics configured for evalset {eval_set.eval_set_id!r}:"
            ' its evaluation criteria are empty, so nothing would be scored'
            ' and the result could not mean anything. Add criteria to the'
            ' evalset\'s test_config.json, or pass --config-file-path.',
            file=sys.stderr,
        )
        return EXIT_ERROR

    for eval_set, _ in eval_sets:
      duplicate_case_ids = _find_duplicate_eval_case_ids(eval_set)
      if duplicate_case_ids:
        # An EvalSet is schema-valid with repeated eval_ids, but
        # add_eval_case() raises on the second one -- and the registration
        # loop below is not inside an error handler, so this would otherwise
        # abort with a traceback and exit 1, the metric-failure code.
        print(
            f"Duplicate eval_id(s) in evalset {eval_set.eval_set_id!r}:"
            f' {", ".join(repr(i) for i in duplicate_case_ids)}. Eval case IDs'
            ' must be unique within an evalset.',
            file=sys.stderr,
        )
        return EXIT_ERROR

    # Metric registration and its validation run here, before any inference:
    # an EvalMetric only carries a `custom_function_path`, and nothing teaches
    # the registry that the metric exists, so an unregistered or unimportable
    # custom metric would otherwise surface during scoring -- after the
    # deployed agent's tools have already had their real-world side effects.
    try:
      check_custom_metrics_are_consistent([
          (eval_set.eval_set_id, eval_config)
          for eval_set, eval_config in eval_sets
      ])
      # One registry for the run, because one RemoteEvalService scores every
      # evalset. The consistency check above is what makes that union
      # faithful to each individual config.
      metric_evaluator_registry = build_metric_evaluator_registry(
          *(eval_config for _, eval_config in eval_sets)
      )
      for eval_set, eval_config in eval_sets:
        check_criteria_have_evaluators(
            metric_evaluator_registry,
            eval_config,
            eval_set_id=eval_set.eval_set_id,
        )
    except ValueError as e:
      print(str(e), file=sys.stderr)
      return EXIT_ERROR
    # No ModuleNotFoundError branch: the registry lives in the very module
    # RemoteEvalService's own guarded import above already pulled in, so if
    # google-adk's eval dependency chain were incomplete this function would
    # have returned with that message long before reaching here.

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
    results_manager = _ReadableNameEvalSetResultsManager(
        agents_dir=args.results_dir
    )
    service = RemoteEvalService(
        client,
        app_name=app_name,
        eval_sets_manager=eval_sets_manager,
        default_user_id=args.user_id,
        keep_sessions=args.keep_sessions,
        metric_evaluator_registry=metric_evaluator_registry,
    )

    had_inference_failure = False
    had_metric_failure = False
    saved_any_results = False
    try:
      for eval_set, eval_config in eval_sets:
        outcome = await _run_and_evaluate_eval_set(
            service,
            results_manager,
            app_name=app_name,
            eval_set=eval_set,
            eval_config=eval_config,
            num_runs=args.num_runs,
            parallelism=args.parallelism,
            print_detailed_results=args.print_detailed_results,
        )
        had_inference_failure = (
            had_inference_failure or outcome.had_inference_failure
        )
        had_metric_failure = had_metric_failure or outcome.had_metric_failure
        saved_any_results = saved_any_results or outcome.saved_results
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

  if saved_any_results:
    eval_history_dir = (
        Path(args.results_dir) / app_name / _ADK_EVAL_HISTORY_SUBDIR
    )
    print(f'Eval results saved under: {eval_history_dir}')
  else:
    # Every eval case failed inference, so evaluate() was never called and no
    # result file exists. Printing the path anyway would point users and
    # automation at an artifact that was never written.
    print(
        'No eval results were saved: no eval case produced a scorable'
        ' inference result.',
        file=sys.stderr,
    )

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
  elif ntpath.splitdrive(app_name)[0]:
    # A drive-qualified name such as 'C:' or 'C:outside' contains no
    # separator and is not a dot segment, but Windows path joining treats it
    # as drive-relative and discards the --results-dir prefix entirely
    # (ntpath.join('results', 'C:outside') == 'C:outside'). Checked with
    # ntpath on every platform, not just Windows, so the same evalset and the
    # same remote /list-apps response are accepted or rejected identically
    # everywhere.
    reason = 'must not be drive-qualified'
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


def _eval_cases_sharing_a_pinned_session(
    eval_sets: Sequence[tuple[EvalSet, EvalConfig]],
    *,
    default_user_id: str,
    group_by_pinned_session,
) -> tuple[tuple[str, str], list[str]] | None:
  """Finds eval cases that resolve to one and the same pinned remote session.

  The grouping itself lives in ``eval_service`` and is shared with
  ``RemoteEvalService``, which fails the conflicting cases for direct library
  callers; this only decides how the CLI reports the same finding.

  Args:
      eval_sets: Every loaded ``(EvalSet, EvalConfig)`` pair; one service runs
          them all, so the search spans evalsets rather than staying within
          one.
      default_user_id: ``--user-id``, applied when an eval case does not name
          its own.
      group_by_pinned_session:
          ``eval_service._group_eval_cases_by_pinned_session``, injected so
          this module does not import the (lazily imported) service at module
          scope.

  Returns:
      ``((user_id, session_id), [eval_case_id, ...])`` for the first session
      claimed by more than one eval case, or ``None`` when every pin is
      unique. Eval cases pinning *different* sessions are fine.
  """
  all_eval_cases = [
      eval_case for eval_set, _ in eval_sets for eval_case in eval_set.eval_cases
  ]
  by_session = group_by_pinned_session(all_eval_cases, default_user_id)
  for key, case_ids in by_session.items():
    if len(case_ids) > 1:
      return key, case_ids
  return None


def _find_duplicate_eval_case_ids(eval_set: EvalSet) -> list[str]:
  """Returns ``eval_id``s appearing more than once in one evalset.

  ``InMemoryEvalSetsManager.add_eval_case`` keys cases by
  ``(app_name, eval_set_id, eval_id)`` and raises ``ValueError`` on a repeat,
  so duplicates only collide within a single evalset -- and duplicate
  ``eval_set_id``s are already rejected separately.
  """
  seen: set[str] = set()
  duplicates: list[str] = []
  for eval_case in eval_set.eval_cases:
    if eval_case.eval_id in seen and eval_case.eval_id not in duplicates:
      duplicates.append(eval_case.eval_id)
    seen.add(eval_case.eval_id)
  return duplicates


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


_ANSI_GREEN = '\x1b[32m'
_ANSI_RED = '\x1b[31m'
_ANSI_RESET = '\x1b[0m'


def _enable_windows_vt_processing(stream) -> bool:
  """Turns on ANSI interpretation for ``stream``'s Windows console, if it can.

  A legacy Windows console reports ``isatty() == True`` yet renders raw ANSI
  escapes as mojibake unless ENABLE_VIRTUAL_TERMINAL_PROCESSING has been set
  on its handle, so colorizing is only safe once this succeeds. Modern
  terminals (Windows Terminal, ConEmu, ...) either preset the flag or accept
  it being set here.

  Returns:
      ``True`` if the console interprets ANSI sequences (the flag was already
      set or was set successfully), ``False`` otherwise -- including on
      non-Windows platforms, where the required modules do not exist.
  """
  try:
    import ctypes
    import msvcrt

    handle = msvcrt.get_osfhandle(stream.fileno())
    kernel32 = ctypes.windll.kernel32
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
      return False
    enable_virtual_terminal_processing = 0x0004
    if mode.value & enable_virtual_terminal_processing:
      return True
    return bool(
        kernel32.SetConsoleMode(
            handle, mode.value | enable_virtual_terminal_processing
        )
    )
  except Exception:  # noqa: BLE001 - any failure means "don't emit ANSI"
    return False


def _colorize(text: str, color: str, *, stream) -> str:
  """Wraps ``text`` in an ANSI color code if ``stream`` looks like a color-capable tty.

  Colorizing is skipped (returning ``text`` unchanged) when ``stream`` is not a
  tty (e.g. piped output, or pytest's ``capsys``), when ``NO_COLOR`` is set
  (see https://no-color.org), when ``TERM=dumb``, or on a Windows console
  whose virtual-terminal processing cannot be enabled (a legacy console shows
  literal escape characters otherwise). No new dependency is introduced for
  this -- raw ANSI escape codes are used directly.

  Args:
      text: The text to colorize.
      color: One of the ``_ANSI_*`` color constants.
      stream: The stream ``text`` is about to be printed to; its ``isatty()``
          is what gates colorization, since stdout and stderr can differ.
  """
  isatty = getattr(stream, 'isatty', None)
  if not callable(isatty) or not isatty():
    return text
  if os.environ.get('NO_COLOR'):
    return text
  if os.environ.get('TERM') == 'dumb':
    return text
  if os.name == 'nt' and not _enable_windows_vt_processing(stream):
    return text
  return f'{color}{text}{_ANSI_RESET}'


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
) -> _EvalSetOutcome:
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
      An :class:`_EvalSetOutcome` for this eval set. ``saved_results`` is
      false when no inference succeeded, since ``evaluate()`` is then never
      called and no result file is written.
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
      prefix = _colorize('INFERENCE FAILED', _ANSI_RED, stream=sys.stderr)
      print(
          f"{prefix} for eval set '{eval_set.eval_set_id}' eval case"
          f" '{inference_result.eval_case_id}': {inference_result.error_message}",
          file=sys.stderr,
      )

  had_metric_failure = False
  saved_results = False
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
    saved_results = True
    for eval_case_result in eval_case_results:
      if _print_eval_case_result(
          eval_case_result, print_detailed_results=print_detailed_results
      ):
        had_metric_failure = True

    if print_detailed_results:
      # The one-line results above are per EvalCaseResult, so with --num-runs N
      # each metric is reported N times. The breakdown tables are per eval
      # *case* instead: every run of one eval id contributes its invocations to
      # a single table, matching how ADK's own AgentEvaluator groups them (and
      # avoiding N copies of the same table). defaultdict keeps insertion
      # order, so cases are reported in the order evaluate() yielded them.
      results_by_eval_id: dict[str, list[EvalCaseResult]] = (
          collections.defaultdict(list)
      )
      for eval_case_result in eval_case_results:
        results_by_eval_id[eval_case_result.eval_id].append(eval_case_result)
      for eval_id, results_per_eval_id in results_by_eval_id.items():
        _print_detailed_metric_results(eval_id, results_per_eval_id)

  return _EvalSetOutcome(
      had_inference_failure=had_inference_failure,
      had_metric_failure=had_metric_failure,
      saved_results=saved_results,
  )


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
    target_stream = sys.stdout if metric_passed else sys.stderr
    color = _ANSI_GREEN if metric_passed else _ANSI_RED
    status_label = _colorize(status_label, color, stream=target_stream)
    print(
        f'[{eval_case_result.eval_id}] {metric_result.metric_name}:'
        f' score={metric_result.score} threshold={metric_result.threshold}'
        f' status={status_label}',
        file=target_stream,
    )
  return case_failed


def _non_passing_metric_results(
    eval_case_results: list[EvalCaseResult],
) -> dict[str, EvalMetricResult]:
  """Returns the first non-passing overall result for each metric, by name.

  A metric is reported as not passing if *any* run of the eval case scored it
  as anything other than ``PASSED`` -- the same metric can pass on one run and
  fail on another, and the breakdown exists to explain exactly that.

  Args:
      eval_case_results: Every run's result for one eval case (one eval id).

  Returns:
      ``{metric_name: EvalMetricResult}`` for the metrics worth a breakdown,
      in the order the metrics were scored. The value is the first non-passing
      result seen, whose ``score``/``threshold`` the summary line reports.
  """
  non_passing: dict[str, EvalMetricResult] = {}
  for eval_case_result in eval_case_results:
    for metric_result in eval_case_result.overall_eval_metric_results:
      if metric_result.eval_status == EvalStatus.PASSED:
        continue
      non_passing.setdefault(metric_result.metric_name, metric_result)
  return non_passing


def _print_detailed_metric_results(
    eval_id: str, eval_case_results: list[EvalCaseResult]
) -> None:
  """Prints a per-invocation breakdown table for one eval case's failures.

  This is what makes ``--print-detailed-results`` live up to its name: the
  one-line results say a metric scored below its threshold, this says what the
  agent actually did. Only metrics that did not pass get a table -- a passing
  metric has nothing to diagnose, and tabulating every invocation of every
  metric would grow with ``--num-runs`` for no benefit.

  Everything is written to stderr, alongside the one-line failures, so that a
  failing run's diagnostics stay on one stream.

  This deliberately does not reuse ``AgentEvaluator._print_details()`` even
  though the table mirrors its columns. That helper writes to stdout via a
  bare ``print()``, which would split one failure's reporting across both
  streams, and it re-derives its own overall score as the mean of the
  per-invocation scores. The score/threshold printed here come from the same
  ``overall_eval_metric_results`` entry the one-line result above used, so the
  breakdown never contradicts the verdict the exit code is derived from.

  Args:
      eval_id: The eval case id these results belong to.
      eval_case_results: Every run's result for that eval case.
  """
  non_passing = _non_passing_metric_results(eval_case_results)
  if not non_passing:
    return

  # Imported here rather than at module scope for the same reason
  # RemoteEvalService is (see this module's docstring): keeping the import off
  # the module path means `pytest-adk eval --help` still works in an
  # environment where the dependency is somehow missing. pytest-adk depends on
  # tabulate directly (see pyproject.toml) rather than relying on google-adk's
  # `eval` extra, so under a normal install this is always available.
  from tabulate import tabulate

  results_with_invocation = (
      _AdkAgentEvaluator._get_eval_metric_results_with_invocation(
          eval_case_results
      )
  )
  for metric_name, metric_result in non_passing.items():
    per_invocation = results_with_invocation.get(metric_name)
    if not per_invocation:
      # Defensive: the metric produced an overall verdict but no
      # per-invocation results, leaving nothing to tabulate. Note a metric
      # evaluator that *raised* is not this case -- google-adk still records a
      # placeholder row per invocation (with no score) alongside the
      # NOT_EVALUATED verdict, and that breakdown is worth showing. The
      # one-line result has already reported the metric either way.
      continue
    status_label = _colorize(
        metric_result.eval_status.name, _ANSI_RED, stream=sys.stderr
    )
    print(
        f'Detail for [{eval_id}] {metric_name}:'
        f' score={metric_result.score} threshold={metric_result.threshold}'
        f' status={status_label}',
        file=sys.stderr,
    )
    rows: list[dict[str, object]] = []
    for result in per_invocation:
      # expected_invocation is optional (a conversation_scenario-driven eval
      # case has no golden turns), so fall back to the actual invocation for
      # the prompt and leave the expected columns empty, as ADK does.
      expected = result.expected_invocation
      rows.append({
          'eval_status': result.eval_metric_result.eval_status.name,
          'score': result.eval_metric_result.score,
          'threshold': metric_result.threshold,
          'prompt': _AdkAgentEvaluator._convert_content_to_text(
              expected.user_content
              if expected
              else result.actual_invocation.user_content
          ),
          'expected_response': _AdkAgentEvaluator._convert_content_to_text(
              expected.final_response if expected else None
          ),
          'actual_response': _AdkAgentEvaluator._convert_content_to_text(
              result.actual_invocation.final_response
          ),
          'expected_tool_calls': (
              _AdkAgentEvaluator._convert_tool_calls_to_text(
                  expected.intermediate_data if expected else None
              )
          ),
          'actual_tool_calls': _AdkAgentEvaluator._convert_tool_calls_to_text(
              result.actual_invocation.intermediate_data
          ),
      })
    print(
        tabulate(rows, headers='keys', tablefmt='grid', maxcolwidths=25),
        file=sys.stderr,
    )


if __name__ == '__main__':
  sys.exit(main())
