# Copyright 2026 pytest-adk contributors
"""Shared pytest configuration for the test suite.

``pytester`` is enabled so plugin-level behavior (e.g. the
``pytest_adk_prompt_template_engine`` ini option) can be exercised end to end in
an isolated, temporary pytest project.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

pytest_plugins = ['pytester']

# A stand-in for the metric module an evaluated project would ship next to its
# evalsets. ``sync_metric`` scores above and ``async_metric`` below the 0.5
# threshold the tests configure, so one run can show a custom metric passing
# and the other failing -- and both call shapes ADK supports
# (``_CustomMetricEvaluator`` awaits coroutine functions and calls plain ones)
# are covered.
_METRIC_MODULE_SOURCE = '''\
"""Project-local custom eval metrics."""

from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult


def _scored(actual_invocations, score):
  status = EvalStatus.PASSED if score >= 0.5 else EvalStatus.FAILED
  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=[
          PerInvocationResult(
              actual_invocation=invocation, score=score, eval_status=status
          )
          for invocation in actual_invocations
      ],
  )


def sync_metric(
    eval_metric, actual_invocations, expected_invocations, conversation_scenario
):
  return _scored(actual_invocations, 0.75)


async def async_metric(
    eval_metric, actual_invocations, expected_invocations, conversation_scenario
):
  return _scored(actual_invocations, 0.25)


not_a_function = 'a module attribute that cannot be called'
'''

_metric_module_counter = itertools.count()


@pytest.fixture
def write_metric_module(tmp_path):
  """Returns a callable that writes a project-local custom metric module.

  The module is *not* put on ``sys.path``: making it importable is exactly
  what the ``--pythonpath`` flag and the working-directory behavior under test
  are responsible for. Tests that only need the module importable can pass its
  directory to ``monkeypatch.syspath_prepend``.

  Each call gets a process-unique module name so that tests cannot observe
  each other's ``sys.modules`` entry, and those entries are removed on
  teardown -- otherwise a module written into one test's ``tmp_path`` would
  keep resolving after that directory is gone.

  Returns:
      ``write(directory=None) -> str``, writing the module into ``directory``
      (default: ``tmp_path``) and returning its importable module name.
  """
  written: list[str] = []

  def write(directory: str | Path | None = None) -> str:
    target = Path(directory) if directory is not None else tmp_path
    target.mkdir(parents=True, exist_ok=True)
    module_name = f'pytest_adk_project_metrics_{next(_metric_module_counter)}'
    (target / f'{module_name}.py').write_text(
        _METRIC_MODULE_SOURCE, encoding='utf-8'
    )
    written.append(module_name)
    return module_name

  yield write

  for module_name in written:
    sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _restore_metric_evaluator_registry():
  """Undoes custom-metric registrations, which ADK keeps in class-level state.

  ``MetricEvaluatorRegistry._registry`` is a *class* attribute (verified in
  google-adk 1.30 through 2.6), so every registry instance -- including the
  process-wide ``DEFAULT_METRIC_EVALUATOR_REGISTRY`` that the fixture path
  registers into -- shares one dict. Without restoring it, a test that
  registers a custom metric would change how every later test resolves that
  metric name.

  Autouse rather than opt-in because the leak happens wherever an eval runs,
  not only in the tests that are about custom metrics.
  """
  try:
    from google.adk.evaluation.metric_evaluator_registry import (
        MetricEvaluatorRegistry,
    )
  except ModuleNotFoundError:
    # google-adk v2 without google-cloud-aiplatform: nothing here can have
    # registered anything either, so there is nothing to restore.
    yield
    return

  snapshot = dict(MetricEvaluatorRegistry._registry)
  yield
  MetricEvaluatorRegistry._registry.clear()
  MetricEvaluatorRegistry._registry.update(snapshot)
