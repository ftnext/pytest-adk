# Copyright 2026 pytest-adk contributors
"""pytest plugin exposing the AgentEvaluator fixture."""

from __future__ import annotations

import pytest

from .evaluation import _AgentEvaluator


@pytest.fixture
def AgentEvaluator(tmp_path):  # noqa: N802 (fixture deliberately named like a class)
  """Return an evaluator bound to pytest's ``tmp_path`` as the results dir.

  Eval result JSON files are written under
  ``tmp_path/test_app/.adk/eval_history/``.
  """
  return _AgentEvaluator(results_dir=tmp_path)
