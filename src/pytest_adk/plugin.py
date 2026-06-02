# Copyright 2026 pytest-adk contributors
"""pytest plugin exposing the AgentEvaluator fixture."""

from __future__ import annotations

import pytest

from .evaluation import _AgentEvaluator

# Session-wide map of test node id -> eval_history directory, populated as each
# ``AgentEvaluator`` fixture tears down and rendered by ``pytest_terminal_summary``.
_EVAL_RESULTS_DIRS = pytest.StashKey[dict]()


@pytest.fixture
def AgentEvaluator(request, tmp_path):  # noqa: N802 (fixture deliberately named like a class)
  """Return an evaluator bound to pytest's ``tmp_path`` as the results dir.

  Eval result JSON files are written under
  ``tmp_path/test_app/.adk/eval_history/``. On teardown, that directory is
  recorded so ``pytest_terminal_summary`` can always report where results landed.
  """
  evaluator = _AgentEvaluator(results_dir=tmp_path)
  yield evaluator
  eval_history_dir = evaluator.eval_history_dir
  if eval_history_dir.exists():
    store = request.config.stash.setdefault(_EVAL_RESULTS_DIRS, {})
    store[request.node.nodeid] = eval_history_dir


def pytest_terminal_summary(terminalreporter, exitstatus, config):
  """Report, per test, where ADK eval results were saved.

  Runs regardless of test outcome, so the results location is always visible.
  """
  store = config.stash.get(_EVAL_RESULTS_DIRS, {})
  if not store:
    return
  terminalreporter.write_sep('=', 'ADK eval results')
  for nodeid, eval_history_dir in store.items():
    terminalreporter.write_line(nodeid)
    terminalreporter.write_line(f'  {eval_history_dir}')
