# Copyright 2026 pytest-adk contributors
"""pytest plugin exposing the AgentEvaluator fixture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from .evaluation import _AgentEvaluator
from .prompt_template import _VALID_ENGINES

if TYPE_CHECKING:
  from _pytest.config import Config
  from _pytest.config.argparsing import Parser
  from _pytest.fixtures import FixtureRequest
  from _pytest.terminal import TerminalReporter

# Session-wide map of test node id -> eval_history directory, populated as each
# ``AgentEvaluator`` fixture tears down and rendered by ``pytest_terminal_summary``.
_EVAL_RESULTS_DIRS = pytest.StashKey[dict[str, Path]]()

# Ini option name for selecting the prompt-template rendering engine. Set it in
# ``pyproject.toml`` under ``[tool.pytest.ini_options]``.
_PROMPT_TEMPLATE_ENGINE_INI = 'pytest_adk_prompt_template_engine'


def pytest_addoption(parser: Parser) -> None:
  """Register the ``pytest_adk_prompt_template_engine`` ini option.

  Opting into Jinja is configured in ``pyproject.toml``::

      [tool.pytest.ini_options]
      pytest_adk_prompt_template_engine = "jinja"
  """
  parser.addini(
      _PROMPT_TEMPLATE_ENGINE_INI,
      help=(
          "Engine for rendering <prompt:...> markers: 'string' (default,"
          " string.Template) or 'jinja' (requires the 'jinja' extra)."
      ),
      default='string',
  )


@pytest.fixture
def AgentEvaluator(  # noqa: N802 (fixture deliberately named like a class)
    request: FixtureRequest,
    tmp_path: Path,
) -> Iterator[_AgentEvaluator]:
  """Return an evaluator bound to pytest's ``tmp_path`` as the results dir.

  Eval result JSON files are written under
  ``tmp_path/test_app/.adk/eval_history/``. On teardown, that directory is
  recorded so ``pytest_terminal_summary`` can always report where results landed.

  The prompt-template engine is read from the
  ``pytest_adk_prompt_template_engine`` ini option (default ``'string'``).
  """
  engine = request.config.getini(_PROMPT_TEMPLATE_ENGINE_INI)
  if engine not in _VALID_ENGINES:
    raise pytest.UsageError(
        f'Invalid {_PROMPT_TEMPLATE_ENGINE_INI} value {engine!r};'
        f' expected one of {", ".join(_VALID_ENGINES)}.'
    )
  evaluator = _AgentEvaluator(
      results_dir=tmp_path, prompt_template_engine=engine
  )
  yield evaluator
  eval_history_dir = evaluator.eval_history_dir
  if eval_history_dir.exists():
    store = request.config.stash.setdefault(_EVAL_RESULTS_DIRS, {})
    store[request.node.nodeid] = eval_history_dir


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,
    config: Config,
) -> None:
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
