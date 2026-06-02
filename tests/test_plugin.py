# Copyright 2026 pytest-adk contributors

from __future__ import annotations

from pathlib import Path

import pytest

from pytest_adk.evaluation import _AgentEvaluator
from pytest_adk.plugin import _EVAL_RESULTS_DIRS
from pytest_adk.plugin import pytest_terminal_summary


def test_eval_history_dir_matches_adk_layout(tmp_path) -> None:
  evaluator = _AgentEvaluator(results_dir=tmp_path)

  # Mirrors the layout asserted by tests/test_evaluation.py::_saved_result_files.
  assert evaluator.eval_history_dir == (
      tmp_path / 'test_app' / '.adk' / 'eval_history'
  )


class _StubTerminalReporter:
  def __init__(self) -> None:
    self.seps: list[tuple[str, str]] = []
    self.lines: list[str] = []

  def write_sep(self, sep: str, title: str, **kwargs) -> None:
    self.seps.append((sep, title))

  def write_line(self, line: str, **kwargs) -> None:
    self.lines.append(line)


class _StubConfig:
  def __init__(self) -> None:
    self.stash = pytest.Stash()


def test_terminal_summary_reports_recorded_dirs() -> None:
  config = _StubConfig()
  history_dir = Path('/tmp/results/test_app/.adk/eval_history')
  config.stash[_EVAL_RESULTS_DIRS] = {'tests/test_x.py::test_home': history_dir}
  reporter = _StubTerminalReporter()

  pytest_terminal_summary(reporter, exitstatus=0, config=config)

  assert ('=', 'ADK eval results') in reporter.seps
  assert 'tests/test_x.py::test_home' in reporter.lines
  assert f'  {history_dir}' in reporter.lines


def test_terminal_summary_silent_when_nothing_recorded() -> None:
  config = _StubConfig()
  reporter = _StubTerminalReporter()

  pytest_terminal_summary(reporter, exitstatus=0, config=config)

  assert reporter.seps == []
  assert reporter.lines == []
