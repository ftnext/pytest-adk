# Copyright 2026 pytest-adk contributors
"""Registration of ``EvalConfig`` custom metrics with ADK's metric registry.

``get_eval_metrics_from_config()`` turns a ``test_config.json`` entry into an
``EvalMetric`` carrying ``custom_function_path``, but it does *not* teach the
:class:`~google.adk.evaluation.metric_evaluator_registry.MetricEvaluatorRegistry`
that the metric exists. Without the registration this module performs, scoring
raises ``NotFoundError: <metric_name> not found in registry`` -- and only
*after* inference has already run, i.e. after a deployed agent's tools have
had their real-world side effects. Hence the validation helpers here, which
resolve every configured metric function up front.

ADK's own ``adk eval`` command does the same registration inline (see
``google/adk/cli/cli_tools_click.py``); this module exists so the
``AgentEvaluator`` pytest fixture and ``pytest-adk eval`` share one
implementation instead of each growing their own copy.

Import constraint: ``google.adk.evaluation.metric_evaluator_registry`` is
never imported at module scope here, only inside the two functions that need
a registry object. On google-adk v2 its import chain requires ``vertexai``
(google-cloud-aiplatform) as a side effect of building the default registry --
the same dependency documented in :mod:`pytest_adk.remote.eval_service` -- and
this module has to stay importable without it so that merely *loading* it
costs nothing extra. The other google-adk imports below were verified
(google-adk 1.30.0 through 2.6.1) to be free of that chain.
"""

from __future__ import annotations

import importlib
from typing import Any
from typing import Callable
from typing import Sequence
from typing import TYPE_CHECKING

from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.eval_config import CustomMetricConfig
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo

if TYPE_CHECKING:
  from google.adk.evaluation.metric_evaluator_registry import (
      MetricEvaluatorRegistry,
  )

# Score range assumed for a custom metric whose config supplies no
# ``metric_info``. Mirrors ADK's own ``cli_eval.get_default_metric_info()``,
# which is not imported because it lives in ``google.adk.cli`` and would drag
# the whole FastAPI-based CLI stack in for one two-line helper.
_DEFAULT_MIN_SCORE = 0.0
_DEFAULT_MAX_SCORE = 1.0

_MISSING_EVAL_DEPENDENCIES_MESSAGE = (
    'Registering custom eval metrics needs'
    " google.adk.evaluation.metric_evaluator_registry, which requires"
    ' google-adk\'s `eval` extra: `pip install "google-adk[eval]"`. On'
    " google-adk v2 that module's import chain additionally requires"
    ' `vertexai` (google-cloud-aiplatform) purely as an import-time side'
    " effect of building the default registry. If pytest-adk's own"
    ' dependencies are already installed, adding the base'
    ' `google-cloud-aiplatform` package on top is therefore sufficient'
    ' without pulling in the rest of the `eval` extra.'
)


def _new_default_registry() -> MetricEvaluatorRegistry:
  """Returns a registry pre-populated with ADK's built-in metric evaluators.

  Raises:
      ModuleNotFoundError: With :data:`_MISSING_EVAL_DEPENDENCIES_MESSAGE`
          when the registry's import chain is unavailable, mirroring
          ``google.adk.evaluation.agent_evaluator``'s own
          ``MISSING_EVAL_DEPENDENCIES_MESSAGE`` treatment rather than letting
          a bare ``No module named 'vertexai'`` escape.
  """
  try:
    from google.adk.evaluation.metric_evaluator_registry import (
        _get_default_metric_evaluator_registry,
    )
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(_MISSING_EVAL_DEPENDENCIES_MESSAGE) from e
  return _get_default_metric_evaluator_registry()


def default_metric_evaluator_registry() -> MetricEvaluatorRegistry:
  """Returns the process-wide registry ADK falls back to when given none.

  ``AgentEvaluator`` (and therefore the ``AgentEvaluator`` pytest fixture)
  constructs its ``LocalEvalService`` without a ``metric_evaluator_registry``,
  so this is the only registry that path can be taught about custom metrics --
  the same object ADK's own ``adk eval`` command registers into.

  Raises:
      ModuleNotFoundError: See :func:`_new_default_registry`.
  """
  try:
    from google.adk.evaluation.metric_evaluator_registry import (
        DEFAULT_METRIC_EVALUATOR_REGISTRY,
    )
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(_MISSING_EVAL_DEPENDENCIES_MESSAGE) from e
  return DEFAULT_METRIC_EVALUATOR_REGISTRY


def build_metric_evaluator_registry(
    *eval_configs: EvalConfig,
) -> MetricEvaluatorRegistry:
  """Returns ADK's default registry plus every config's custom metrics.

  Args:
      *eval_configs: The ``EvalConfig``s whose ``custom_metrics`` should be
          registered. Passing several is how ``pytest-adk eval`` serves the
          one ``RemoteEvalService`` it builds for a run that loaded more than
          one evalset; :func:`check_custom_metrics_are_consistent` is what
          keeps that union faithful to each individual config.

  Returns:
      A registry usable as ``LocalEvalService``'s ``metric_evaluator_registry``.

  Raises:
      ValueError: If any configured custom metric cannot be resolved. See
          :func:`register_custom_metrics`.
  """
  registry = _new_default_registry()
  for eval_config in eval_configs:
    register_custom_metrics(registry, eval_config)
  return registry


def register_custom_metrics(
    registry: MetricEvaluatorRegistry, eval_config: EvalConfig
) -> None:
  """Registers ``eval_config``'s custom metrics on ``registry``.

  Every configured metric function is imported first, so an unimportable
  module or a misspelled function name fails here -- before any inference runs
  -- rather than during scoring, when a deployed agent's tools have already
  had their side effects.

  Args:
      registry: The registry to register into. Mutated in place.
      eval_config: Config whose ``custom_metrics`` entries are registered.

  Raises:
      ValueError: If any configured custom metric function cannot be resolved,
          with one ready-to-print line per offending metric.
  """
  custom_metrics = eval_config.custom_metrics or {}
  if not custom_metrics:
    return

  errors: list[str] = []
  for metric_name, config in custom_metrics.items():
    try:
      _resolve_metric_function(metric_name, config)
    except ValueError as e:
      errors.append(str(e))
  if errors:
    raise ValueError('\n'.join(errors))

  for metric_name, config in custom_metrics.items():
    registry.register_evaluator(
        metric_info=custom_metric_info(metric_name, config),
        evaluator=_CustomMetricEvaluator,
    )


def check_criteria_have_evaluators(
    registry: MetricEvaluatorRegistry,
    eval_config: EvalConfig,
    *,
    eval_set_id: str,
) -> None:
  """Verifies every metric in ``criteria`` has an evaluator in ``registry``.

  ``MetricEvaluatorRegistry.get_evaluator()`` raises ``NotFoundError`` for an
  unknown metric name, but only once scoring reaches that metric -- after
  inference. Checking the whole criteria set up front turns a misspelled
  metric name into an error the caller can report before contacting the agent.

  Args:
      registry: The registry that will score this evalset.
      eval_config: The evalset's config.
      eval_set_id: Used only to name the offending evalset in the message.

  Raises:
      ValueError: If a criteria metric has no registered evaluator.
  """
  registered = {
      metric_info.metric_name
      for metric_info in registry.get_registered_metrics()
  }
  unknown = [
      eval_metric.metric_name
      for eval_metric in get_eval_metrics_from_config(eval_config)
      if eval_metric.metric_name not in registered
  ]
  if not unknown:
    return
  configured = sorted(eval_config.custom_metrics or {})
  declares = (
      ', '.join(repr(i) for i in configured) if configured else 'none'
  )
  raise ValueError(
      f'Evaluation criteria for evalset {eval_set_id!r} name metric(s) with no'
      f' evaluator: {", ".join(repr(i) for i in unknown)}. Use one of ADK\'s'
      f' built-in metrics ({", ".join(sorted(registered))}), or declare the'
      ' metric under `custom_metrics` in the same config (which currently'
      f' declares {declares}).'
  )


def check_custom_metrics_are_consistent(
    labelled_configs: Sequence[tuple[str, EvalConfig]],
) -> None:
  """Verifies several configs can share one registry without changing meaning.

  A registry maps a metric *name* to one evaluator, so when a single run loads
  several evalsets their configs must agree about what each name means. Two
  disagreements are possible, and both would score an evalset with an
  evaluator its own config never asked for:

  * the same name declared as a custom metric by two configs, but pointing at
    different functions (or described by different ``metric_info``);
  * a name declared as a custom metric by one config while another config uses
    that very name in ``criteria`` without declaring it -- typically a
    built-in metric that the first config shadows.

  Both are rejected rather than silently resolved, because either outcome is a
  score for something other than what the evalset describes.

  Args:
      labelled_configs: ``(eval_set_id, EvalConfig)`` pairs for one run.

  Raises:
      ValueError: On the first disagreement found.
  """
  # metric name -> (eval_set_id, function path, metric info) of the first
  # config that declared it.
  declared: dict[str, tuple[str, str, MetricInfo]] = {}
  for eval_set_id, eval_config in labelled_configs:
    for metric_name, config in (eval_config.custom_metrics or {}).items():
      metric_info = custom_metric_info(metric_name, config)
      function_path = config.code_config.name
      previous = declared.get(metric_name)
      if previous is None:
        declared[metric_name] = (eval_set_id, function_path, metric_info)
        continue
      previous_eval_set_id, previous_path, previous_info = previous
      if previous_path == function_path and previous_info == metric_info:
        continue
      raise ValueError(
          f'Custom metric {metric_name!r} is defined differently by evalsets'
          f' {previous_eval_set_id!r} ({previous_path}) and {eval_set_id!r}'
          f' ({function_path}). One evaluation run shares a single metric'
          ' registry, so one of the two definitions would silently score the'
          ' other evalset. Give the metrics distinct names, make the two'
          ' definitions identical, or evaluate the evalsets in separate runs.'
      )

  for eval_set_id, eval_config in labelled_configs:
    own_custom_metrics = eval_config.custom_metrics or {}
    for eval_metric in get_eval_metrics_from_config(eval_config):
      metric_name = eval_metric.metric_name
      if metric_name in own_custom_metrics or metric_name not in declared:
        continue
      other_eval_set_id, function_path, _ = declared[metric_name]
      raise ValueError(
          f'Evalset {eval_set_id!r} uses metric {metric_name!r} without'
          f' declaring it under `custom_metrics`, but evalset'
          f' {other_eval_set_id!r} declares that name as a custom metric'
          f' ({function_path}). One evaluation run shares a single metric'
          f' registry, so {eval_set_id!r} would be scored with that custom'
          ' function instead of the metric its own config means. Rename the'
          ' custom metric, declare it in both configs, or evaluate the'
          ' evalsets in separate runs.'
      )


def custom_metric_info(
    metric_name: str, config: CustomMetricConfig
) -> MetricInfo:
  """Returns the ``MetricInfo`` a custom metric is registered under.

  A configured ``metric_info`` is honored but deep-copied and renamed: the
  registry keys on ``MetricInfo.metric_name``, so a config whose nested
  ``metric_name`` disagrees with its ``custom_metrics`` key would otherwise
  register the evaluator under a name nothing looks up. Copying keeps the
  caller's ``EvalConfig`` unmutated.

  Otherwise a default is built from ``description`` and a ``[0.0, 1.0]`` score
  interval, matching ADK's ``cli_eval.get_default_metric_info()``.
  """
  if config.metric_info is not None:
    metric_info = config.metric_info.model_copy(deep=True)
    metric_info.metric_name = metric_name
    return metric_info
  return MetricInfo(
      metric_name=metric_name,
      description=config.description,
      metric_value_info=MetricValueInfo(
          interval=Interval(
              min_value=_DEFAULT_MIN_SCORE, max_value=_DEFAULT_MAX_SCORE
          )
      ),
  )


def _resolve_metric_function(
    metric_name: str, config: CustomMetricConfig
) -> Callable[..., Any]:
  """Imports a configured custom metric function, or explains why it cannot.

  Deliberately more granular than ADK's own
  ``custom_metric_evaluator._get_metric_function()``, which collapses every
  cause into one ``Could not import ...`` message: the three failures below
  call for different fixes (a ``--pythonpath``/packaging problem, a typo, and
  a wrong attribute), and this runs early precisely so the user can act on it.

  Both sync and async functions are accepted;
  ``_CustomMetricEvaluator.evaluate_invocations`` awaits the coroutine
  functions and calls the plain ones.

  Returns:
      The resolved function object (also warms ``sys.modules``, so the
      evaluator's own later lookup cannot fail differently).

  Raises:
      ValueError: With a ready-to-print message if resolution fails.
  """
  function_path = config.code_config.name
  module_name, _, function_name = function_path.rpartition('.')
  if not module_name or not function_name:
    raise ValueError(
        f'Custom metric {metric_name!r} has an unusable function path'
        f' {function_path!r}: expected a fully qualified'
        " 'my_package.my_module.my_function'."
    )

  try:
    module = importlib.import_module(module_name)
  except Exception as e:  # noqa: BLE001 - a metric module may fail on anything
    # Not just ModuleNotFoundError: the module may exist and raise while
    # being executed. Either way the metric cannot run, and the cause belongs
    # in the message rather than in a traceback from the middle of scoring.
    raise ValueError(
        f'Custom metric {metric_name!r} could not be loaded: importing module'
        f' {module_name!r} (from function path {function_path!r}) failed with'
        f' {type(e).__name__}: {e}. Project-local metric modules must be'
        ' importable from the working directory the command runs in, or from'
        ' a directory passed with --pythonpath.'
    ) from e

  try:
    metric_function = getattr(module, function_name)
  except AttributeError as e:
    raise ValueError(
        f'Custom metric {metric_name!r} could not be loaded: module'
        f' {module_name!r} has no attribute {function_name!r} (from function'
        f' path {function_path!r}).'
    ) from e

  if not callable(metric_function):
    raise ValueError(
        f'Custom metric {metric_name!r} resolves to {function_path!r}, which'
        f' is not callable (it is a {type(metric_function).__name__}). Point'
        ' it at the metric function itself.'
    )
  return metric_function
