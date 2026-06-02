# Copyright 2026 pytest-adk contributors

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.genai import types

import pytest_adk.evaluation as evaluation_module
from pytest_adk import AgentEvaluator


def _eval_case_result(eval_id: str, run_index: int = 0) -> EvalCaseResult:
  invocation = Invocation(
      userContent=types.Content(
          role='user',
          parts=[types.Part(text=f'prompt {run_index}')],
      ),
      finalResponse=types.Content(
          role='model',
          parts=[types.Part(text=f'response {run_index}')],
      ),
  )
  metric_result = EvalMetricResult(
      metricName='test_metric',
      threshold=0.5,
      score=1.0,
      evalStatus=EvalStatus.PASSED,
  )
  return EvalCaseResult(
      evalSetId='placeholder',
      evalId=eval_id,
      finalEvalStatus=EvalStatus.PASSED,
      overallEvalMetricResults=[metric_result],
      evalMetricResultPerInvocation=[
          EvalMetricResultPerInvocation(
              actualInvocation=invocation,
              expectedInvocation=invocation,
              evalMetricResults=[metric_result],
          )
      ],
      sessionId=f'session-{eval_id}-{run_index}',
  )


def _patch_successful_adk_eval(monkeypatch, *, seen_test_files=None) -> None:
  config = SimpleNamespace(user_simulator_config=None)
  agent_for_eval = object()
  eval_metrics = [object()]

  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_initial_session',
      staticmethod(lambda initial_session_file=None: {}),
  )
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      'find_config_for_test_file',
      staticmethod(lambda test_file: config),
  )

  def load_eval_set(test_file, eval_config, initial_session):
    if seen_test_files is not None:
      seen_test_files.append(test_file)
    return SimpleNamespace(eval_set_id=Path(test_file).stem, eval_cases=[])

  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_load_eval_set_from_file',
      staticmethod(load_eval_set),
  )

  async def get_agent_for_eval(module_name, agent_name=None):
    return agent_for_eval

  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_agent_for_eval',
      staticmethod(get_agent_for_eval),
  )
  monkeypatch.setattr(
      evaluation_module,
      'get_eval_metrics_from_config',
      lambda eval_config: eval_metrics,
  )
  monkeypatch.setattr(
      evaluation_module,
      'UserSimulatorProvider',
      lambda user_simulator_config: SimpleNamespace(
          user_simulator_config=user_simulator_config
      ),
  )

  async def get_eval_results_by_eval_id(
      agent_for_eval,
      eval_set,
      eval_metrics,
      num_runs,
      user_simulator_provider,
  ):
    return {
        'case-1': [
            _eval_case_result('case-1', run_index)
            for run_index in range(num_runs)
        ]
    }

  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_eval_results_by_eval_id',
      staticmethod(get_eval_results_by_eval_id),
  )
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_eval_metric_results_with_invocation',
      staticmethod(lambda eval_results_per_eval_id: {}),
  )
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_process_metrics_and_get_failures',
      staticmethod(
          lambda eval_metric_results, print_detailed_results, agent_module: []
      ),
  )


def _saved_result_files(results_dir: Path) -> list[Path]:
  return list(
      (results_dir / 'test_app' / '.adk' / 'eval_history').glob(
          '*.evalset_result.json'
      )
  )


@pytest.mark.asyncio
async def test_agent_evaluator_is_exported_and_saves_single_file(
    tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'single.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  result = await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=test_file,
      num_runs=1,
      results_dir=tmp_path,
  )

  assert result is None
  saved_files = _saved_result_files(tmp_path)
  assert len(saved_files) == 1
  saved_result = json.loads(saved_files[0].read_text(encoding='utf-8'))
  assert saved_result['eval_set_id'] == 'single.test'
  assert len(saved_result['eval_case_results']) == 1


@pytest.mark.asyncio
async def test_agent_evaluator_directory_finds_recursive_json_files(
    tmp_path, monkeypatch
) -> None:
  seen_test_files = []
  root_test = tmp_path / 'root.test.json'
  nested = tmp_path / 'nested'
  nested.mkdir()
  nested_test = nested / 'nested.test.json'
  not_json = nested / 'notes.txt'
  for path in [root_test, nested_test, not_json]:
    path.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch, seen_test_files=seen_test_files)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=tmp_path,
      num_runs=1,
      results_dir=tmp_path / 'results',
  )

  assert set(seen_test_files) == {str(root_test), str(nested_test)}
  assert len(_saved_result_files(tmp_path / 'results')) == 2


@pytest.mark.asyncio
async def test_agent_evaluator_processes_json_without_test_suffix_and_warns(
    tmp_path, monkeypatch, caplog
) -> None:
  seen_test_files = []
  plain = tmp_path / 'plain.json'
  plain.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch, seen_test_files=seen_test_files)

  with caplog.at_level('WARNING', logger=evaluation_module.logger.name):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=tmp_path,
        num_runs=1,
        results_dir=tmp_path / 'results',
    )

  assert seen_test_files == [str(plain)]
  assert len(_saved_result_files(tmp_path / 'results')) == 1
  assert any(
      str(plain) in record.getMessage()
      and 'naming convention' in record.getMessage()
      for record in caplog.records
  )


@pytest.mark.asyncio
async def test_agent_evaluator_saves_each_run_for_eval_case(
    tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'multi.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=str(test_file),
      num_runs=3,
      results_dir=tmp_path,
  )

  saved_files = _saved_result_files(tmp_path)
  saved_result = json.loads(saved_files[0].read_text(encoding='utf-8'))
  assert len(saved_result['eval_case_results']) == 3


@pytest.mark.asyncio
async def test_agent_evaluator_saves_before_raising_for_metric_failure(
    tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'failure.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_process_metrics_and_get_failures',
      staticmethod(
          lambda eval_metric_results, print_detailed_results, agent_module: [
              'test_metric for None Failed. Expected 0.5, but got 0.0.'
          ]
      ),
  )

  with pytest.raises(AssertionError, match='Following are all the test failures'):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
        print_detailed_results=False,
        results_dir=tmp_path,
    )

  saved_files = _saved_result_files(tmp_path)
  assert len(saved_files) == 1


_MULTILINE_TOML_EVALSET = '''\
eval_set_id = "home_automation"

[[eval_cases]]
eval_id = "turn_on_living_room"

[[eval_cases.conversation]]
invocation_id = "inv-1"

[eval_cases.conversation.user_content]
role = "user"
parts = [ { text = """
Please turn on the living room light.
Then confirm it is on.
""" } ]

[eval_cases.conversation.final_response]
role = "model"
parts = [ { text = "The living room light is now on." } ]
'''


def test_load_eval_set_from_toml_minimal(tmp_path) -> None:
  test_file = tmp_path / 'minimal.test.toml'
  test_file.write_text(
      'eval_set_id = "x"\neval_cases = []\n', encoding='utf-8'
  )

  eval_set = evaluation_module._load_eval_set_from_toml(test_file)

  assert eval_set.eval_set_id == 'x'
  assert eval_set.eval_cases == []


def test_load_eval_set_from_toml_preserves_multiline_prompt(tmp_path) -> None:
  test_file = tmp_path / 'multiline.test.toml'
  test_file.write_text(_MULTILINE_TOML_EVALSET, encoding='utf-8')

  eval_set = evaluation_module._load_eval_set_from_toml(test_file)

  assert eval_set.eval_set_id == 'home_automation'
  invocation = eval_set.eval_cases[0].conversation[0]
  user_text = invocation.user_content.parts[0].text
  assert (
      user_text
      == 'Please turn on the living room light.\nThen confirm it is on.\n'
  )


@pytest.mark.asyncio
async def test_agent_evaluator_directory_excludes_test_config_json(
    tmp_path, monkeypatch
) -> None:
  seen_test_files = []
  evalset = tmp_path / 'cases.test.json'
  config = tmp_path / 'test_config.json'
  evalset.write_text('{}', encoding='utf-8')
  config.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch, seen_test_files=seen_test_files)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=tmp_path,
      num_runs=1,
      results_dir=tmp_path / 'results',
  )

  assert seen_test_files == [str(evalset)]
  assert len(_saved_result_files(tmp_path / 'results')) == 1


@pytest.mark.asyncio
async def test_agent_evaluator_directory_finds_json_and_toml(
    tmp_path, monkeypatch
) -> None:
  json_test = tmp_path / 'cases.test.json'
  toml_test = tmp_path / 'cases.test.toml'
  not_evalset = tmp_path / 'README.md'
  json_test.write_text('{}', encoding='utf-8')
  toml_test.write_text(_MULTILINE_TOML_EVALSET, encoding='utf-8')
  not_evalset.write_text('# not an evalset', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=tmp_path,
      num_runs=1,
      results_dir=tmp_path / 'results',
  )

  assert len(_saved_result_files(tmp_path / 'results')) == 2


@pytest.mark.asyncio
async def test_agent_evaluator_toml_rejects_initial_session_file(
    tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'cases.test.toml'
  test_file.write_text(_MULTILINE_TOML_EVALSET, encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_initial_session',
      staticmethod(lambda initial_session_file=None: {'state': {'k': 'v'}}),
  )

  with pytest.raises(AssertionError, match='not supported for TOML'):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
        initial_session_file='initial.json',
        results_dir=tmp_path / 'results',
    )
