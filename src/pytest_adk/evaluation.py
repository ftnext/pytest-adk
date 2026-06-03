# Copyright 2026 pytest-adk contributors
"""Pytest-friendly ADK evaluation helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
  import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
  import tomli as tomllib

from google.adk.evaluation import AgentEvaluator as _AdkAgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)
from google.adk.evaluation.simulation.user_simulator_provider import (
    UserSimulatorProvider,
)

from .prompt_template import _expand_prompt_templates

logger = logging.getLogger(__name__)

_EVAL_APP_NAME = 'test_app'
_NUM_RUNS = 2

# Subpath that ADK's LocalEvalSetResultsManager writes ``*.evalset_result.json``
# files into, relative to ``{results_dir}/{app_name}/``.
_ADK_EVAL_HISTORY_SUBDIR = Path('.adk') / 'eval_history'


def _load_eval_set_from_toml(eval_set_file: str | Path) -> EvalSet:
  """Load an EvalSet from a TOML file (new EvalSet schema only).

  Unlike ADK's JSON loader, this does not support the legacy data format or an
  explicit ``initial_session``; the initial session must be expressed inside the
  EvalSet schema.

  ``tomllib.load`` requires a binary handle, so we read text and use
  ``tomllib.loads`` to stay consistent with the rest of the package.

  Args:
      eval_set_file: Path to a TOML file containing the ADK ``EvalSet`` schema.

  Returns:
      The validated ADK ``EvalSet`` model.
  """
  data = tomllib.loads(Path(eval_set_file).read_text(encoding='utf-8'))
  return EvalSet.model_validate(data)


class _AgentEvaluator:
  """ADK AgentEvaluator wrapper that persists local eval results.

  Construct with a ``results_dir``; the ``AgentEvaluator`` pytest fixture binds
  it to pytest's ``tmp_path`` (see :mod:`pytest_adk.plugin`).
  """

  def __init__(
      self,
      results_dir: str | Path,
      prompt_template_engine: str = 'string',
  ) -> None:
    """Create an evaluator that writes ADK eval history under ``results_dir``.

    Args:
        results_dir: Base directory passed to ADK's local eval results manager.
            The pytest fixture supplies pytest's per-test ``tmp_path``.
        prompt_template_engine: Engine used to render ``<prompt:...>`` markers,
            either ``'string'`` (default, ``string.Template``) or ``'jinja'``.
            The pytest fixture supplies the value of the
            ``pytest_adk_prompt_template_engine`` ini option.
    """
    self._results_dir = results_dir
    self._prompt_template_engine = prompt_template_engine

  @property
  def results_dir(self) -> str | Path:
    """Directory under which eval results are saved."""
    return self._results_dir

  @property
  def eval_history_dir(self) -> Path:
    """Directory where ADK writes ``*.evalset_result.json`` files.

    This mirrors the layout used by
    :class:`~google.adk.evaluation.local_eval_set_results_manager.LocalEvalSetResultsManager`
    so the location can be surfaced (e.g. by the ``AgentEvaluator`` plugin's
    terminal summary) without re-deriving the path elsewhere.
    """
    return Path(self._results_dir) / _EVAL_APP_NAME / _ADK_EVAL_HISTORY_SUBDIR

  async def evaluate(
      self,
      agent_module: str,
      eval_dataset_file_path_or_dir: str | Path,
      num_runs: int = _NUM_RUNS,
      agent_name: str | None = None,
      initial_session_file: str | None = None,
      print_detailed_results: bool = True,
  ) -> None:
    """Evaluate an ADK agent and save generated eval results to disk.

    This mirrors :meth:`google.adk.evaluation.AgentEvaluator.evaluate`, with an
    added persistence hook that saves the per-test-file ``EvalSetResult`` under
    the bound ``results_dir`` before metric failures are asserted.

    Example:
        .. code-block:: python

           @pytest.mark.asyncio
           async def test_with_single_test_file(AgentEvaluator):
             await AgentEvaluator.evaluate(
                 agent_module='home_automation_agent',
                 eval_dataset_file_path_or_dir=(
                     'tests/integration/fixture/home_automation_agent/'
                     'simple_test.test.json'
                 ),
             )

        ``AgentEvaluator`` is a pytest fixture (auto-registered via the
        ``pytest11`` entry point) that binds ``results_dir`` to pytest's
        ``tmp_path``. Eval result JSON files are written under
        ``results_dir/test_app/.adk/eval_history/``.

    Background:
        This helper was inspired by the workflow described in:
        https://nikkie-ftnext.hatenablog.com/entry/google-adk-python-evaluation-use-local-eval-set-results-manager
        The upstream ADK PR for optional eval result persistence is still open:
        https://github.com/google/adk-python/pull/4414

    Args:
        agent_module: Import path of the ADK agent module to evaluate.
        eval_dataset_file_path_or_dir: Evalset file, or a directory searched
            recursively for ``*.test.json`` and ``*.test.toml`` files.
        num_runs: Number of ADK evaluation runs per eval case.
        agent_name: Optional agent variable name inside ``agent_module``.
        initial_session_file: Optional initial session file for JSON evalsets.
            TOML evalsets reject this because they support the current
            ``EvalSet`` schema only.
        print_detailed_results: Whether ADK should print detailed metric output.

    Raises:
        AssertionError: If any ADK metric fails, after eval results have been
            saved to disk.
    """
    eval_dataset_path = os.fspath(eval_dataset_file_path_or_dir)
    test_files: list[str] = []
    if os.path.isdir(eval_dataset_path):
      # When a directory is given, only the ADK naming convention
      # (``.test.json`` / ``.test.toml``) is picked up recursively. This keeps
      # sibling files such as ``test_config.json`` (eval metrics) and the
      # ``*.evalset_result.json`` files written by this helper from being
      # mistakenly loaded as evalsets.
      for root, _, files in os.walk(eval_dataset_path):
        for file in files:
          if file.endswith(('.test.json', '.test.toml')):
            test_files.append(os.path.join(root, file))
    else:
      # A directly specified file is taken at face value; the user's intent is
      # explicit. Extension routing (and a naming-convention warning) happens
      # per file below.
      test_files = [eval_dataset_path]

    initial_session = _AdkAgentEvaluator._get_initial_session(
        initial_session_file
    )

    for test_file in test_files:
      # Files discovered via a directory always satisfy the convention, so this
      # only fires for directly specified files that skip the ``.test.`` infix.
      # The check uses the basename so a ``.test.`` directory name does not mask
      # a non-conventional file.
      if '.test.' not in os.path.basename(test_file):
        logger.warning(
            'Evalset file %r does not follow the .test.json/.test.toml naming'
            ' convention; loading it anyway because it was specified directly.',
            test_file,
        )

      eval_config = _AdkAgentEvaluator.find_config_for_test_file(test_file)
      if test_file.endswith('.toml'):
        assert len(initial_session) == 0, (
            'Initial session should be specified as a part of the EvalSet file.'
            ' An explicit initial_session_file is not supported for TOML'
            ' evalsets, which use the EvalSet schema only.'
        )
        eval_set = _load_eval_set_from_toml(test_file)
      else:
        eval_set = _AdkAgentEvaluator._load_eval_set_from_file(
            test_file, eval_config, initial_session
        )

      eval_set = _expand_prompt_templates(
          eval_set, Path(test_file).parent, self._prompt_template_engine
      )

      await self._evaluate_eval_set_and_save(
          agent_module=agent_module,
          eval_set=eval_set,
          eval_config=eval_config,
          num_runs=num_runs,
          agent_name=agent_name,
          print_detailed_results=print_detailed_results,
      )

  async def _evaluate_eval_set_and_save(
      self,
      *,
      agent_module: str,
      eval_set: EvalSet,
      eval_config: EvalConfig,
      num_runs: int,
      agent_name: str | None,
      print_detailed_results: bool,
  ) -> None:
    """Run ADK evaluation for one ``EvalSet``, persist it, then assert metrics."""
    agent_for_eval = await _AdkAgentEvaluator._get_agent_for_eval(
        module_name=agent_module, agent_name=agent_name
    )
    eval_metrics = get_eval_metrics_from_config(eval_config)
    user_simulator_provider = UserSimulatorProvider(
        user_simulator_config=eval_config.user_simulator_config
    )

    eval_results_by_eval_id = (
        await _AdkAgentEvaluator._get_eval_results_by_eval_id(
            agent_for_eval=agent_for_eval,
            eval_set=eval_set,
            eval_metrics=eval_metrics,
            num_runs=num_runs,
            user_simulator_provider=user_simulator_provider,
        )
    )

    results_manager = LocalEvalSetResultsManager(
        agents_dir=os.fspath(self._results_dir)
    )
    all_eval_results: list[EvalCaseResult] = [
        result
        for eval_results_per_eval_id in eval_results_by_eval_id.values()
        for result in eval_results_per_eval_id
    ]
    results_manager.save_eval_set_result(
        app_name=_EVAL_APP_NAME,
        eval_set_id=eval_set.eval_set_id,
        eval_case_results=all_eval_results,
    )

    failures: list[str] = []
    for eval_results_per_eval_id in eval_results_by_eval_id.values():
      eval_metric_results = (
          _AdkAgentEvaluator._get_eval_metric_results_with_invocation(
              eval_results_per_eval_id
          )
      )
      failures_per_eval_case = (
          _AdkAgentEvaluator._process_metrics_and_get_failures(
              eval_metric_results=eval_metric_results,
              print_detailed_results=print_detailed_results,
              agent_module=agent_name,
          )
      )
      failures.extend(failures_per_eval_case)

    failure_message = 'Following are all the test failures.'
    if not print_detailed_results:
      failure_message += (
          ' If you looking to get more details on the failures, then please'
          ' re-run this test with `print_detailed_results` set to `True`.'
      )
    failure_message += '\n' + '\n'.join(failures)
    assert not failures, failure_message
