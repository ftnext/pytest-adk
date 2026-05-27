# Copyright 2026 pytest-adk contributors
"""Pytest-friendly ADK evaluation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from google.adk.evaluation import AgentEvaluator as _AdkAgentEvaluator
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)
from google.adk.evaluation.simulation.user_simulator_provider import (
    UserSimulatorProvider,
)

_EVAL_APP_NAME = 'test_app'
_NUM_RUNS = 2


class AgentEvaluator:
  """ADK AgentEvaluator wrapper that persists local eval results."""

  @staticmethod
  async def evaluate(
      agent_module: str,
      eval_dataset_file_path_or_dir: str | Path,
      num_runs: int = _NUM_RUNS,
      agent_name: Optional[str] = None,
      initial_session_file: Optional[str] = None,
      print_detailed_results: bool = True,
      *,
      results_dir: str | Path,
  ) -> None:
    """Evaluate an ADK agent and save generated eval results to disk.

    This mirrors :meth:`google.adk.evaluation.AgentEvaluator.evaluate`, with an
    added ``results_dir`` hook that saves the per-test-file
    ``EvalSetResult`` before metric failures are asserted.
    """
    eval_dataset_path = os.fspath(eval_dataset_file_path_or_dir)
    test_files = []
    if os.path.isdir(eval_dataset_path):
      for root, _, files in os.walk(eval_dataset_path):
        for file in files:
          if file.endswith('.test.json'):
            test_files.append(os.path.join(root, file))
    else:
      test_files = [eval_dataset_path]

    initial_session = _AdkAgentEvaluator._get_initial_session(
        initial_session_file
    )

    for test_file in test_files:
      eval_config = _AdkAgentEvaluator.find_config_for_test_file(test_file)
      eval_set = _AdkAgentEvaluator._load_eval_set_from_file(
          test_file, eval_config, initial_session
      )

      await AgentEvaluator._evaluate_eval_set_and_save(
          agent_module=agent_module,
          eval_set=eval_set,
          eval_config=eval_config,
          num_runs=num_runs,
          agent_name=agent_name,
          print_detailed_results=print_detailed_results,
          results_dir=results_dir,
      )

  @staticmethod
  async def _evaluate_eval_set_and_save(
      *,
      agent_module: str,
      eval_set,
      eval_config,
      num_runs: int,
      agent_name: Optional[str],
      print_detailed_results: bool,
      results_dir: str | Path,
  ) -> None:
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
        agents_dir=os.fspath(results_dir)
    )
    all_eval_results = [
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
