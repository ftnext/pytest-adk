# Copyright 2026 pytest-adk contributors

from __future__ import annotations

import json
import re
from datetime import datetime
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
  # `custom_metrics=None` mirrors EvalConfig's own default: _AgentEvaluator
  # reads it to decide whether the metric registry needs teaching.
  config = SimpleNamespace(user_simulator_config=None, custom_metrics=None)
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
async def test_agent_evaluator_saves_single_file(
    AgentEvaluator, tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'single.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  result = await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=test_file,
      num_runs=1,
  )

  assert result is None
  saved_files = _saved_result_files(AgentEvaluator.results_dir)
  assert len(saved_files) == 1
  saved_file = saved_files[0]
  saved_stem = saved_file.name.removesuffix('.evalset_result.json')
  assert re.fullmatch(r'test_app_single\.test_\d{8}-\d{6}', saved_stem)
  saved_result = json.loads(saved_file.read_text(encoding='utf-8'))
  assert saved_result['eval_set_id'] == 'single.test'
  assert len(saved_result['eval_case_results']) == 1
  assert saved_result['eval_set_result_id'] == saved_stem
  assert saved_result['eval_set_result_name'] == saved_stem
  assert isinstance(saved_result['creation_timestamp'], float)


def test_same_second_saves_keep_every_result_file(tmp_path, monkeypatch) -> None:
  """Concurrent-style saves in the same local second must not overwrite.

  The datetime is frozen so both saves compute the same ``YYYYMMDD-HHMMSS``
  stem; the second save must claim the ``-2`` name instead of replacing the
  first file.
  """

  class _FrozenDatetime:
    @staticmethod
    def now() -> datetime:
      return datetime(2026, 8, 26, 12, 34, 56)

  monkeypatch.setattr(evaluation_module, 'datetime', _FrozenDatetime)
  manager = evaluation_module._ReadableNameEvalSetResultsManager(
      agents_dir=str(tmp_path)
  )

  manager.save_eval_set_result(
      'test_app', 'single.test', [_eval_case_result('case1')]
  )
  manager.save_eval_set_result(
      'test_app', 'single.test', [_eval_case_result('case2')]
  )

  history_dir = tmp_path / 'test_app' / '.adk' / 'eval_history'
  # iterdir(), not the suffix glob: also proves no leftovers survive the
  # rename (no unix-timestamp originals, no ``.tmp`` intermediates).
  assert sorted(p.name for p in history_dir.iterdir()) == [
      'test_app_single.test_20260826-123456-2.evalset_result.json',
      'test_app_single.test_20260826-123456.evalset_result.json',
  ]
  saved_files = _saved_result_files(tmp_path)
  saved_eval_ids = set()
  for saved_file in saved_files:
    payload = json.loads(saved_file.read_text(encoding='utf-8'))
    stem = saved_file.name.removesuffix('.evalset_result.json')
    assert payload['eval_set_result_id'] == stem
    assert payload['eval_set_result_name'] == stem
    saved_eval_ids.update(
        case['eval_id'] for case in payload['eval_case_results']
    )
  assert saved_eval_ids == {'case1', 'case2'}


def test_save_identifies_its_own_file_despite_concurrent_writers(
    tmp_path, monkeypatch
) -> None:
  """A file landing concurrently in the shared dir must not derail the save.

  The save stages through a private directory, so another writer completing
  mid-save can neither be mistaken for our output nor make our file count
  ambiguous; our result still gets the readable name and the foreign file is
  left alone.
  """

  class _FrozenDatetime:
    @staticmethod
    def now() -> datetime:
      return datetime(2026, 8, 26, 12, 34, 56)

  monkeypatch.setattr(evaluation_module, 'datetime', _FrozenDatetime)
  history_dir = tmp_path / 'test_app' / '.adk' / 'eval_history'
  foreign = history_dir / 'test_app_single.test_1756180000.5.evalset_result.json'
  original_save = evaluation_module.LocalEvalSetResultsManager.save_eval_set_result

  def racing_save(self, *, app_name, eval_set_id, eval_case_results):
    original_save(
        self,
        app_name=app_name,
        eval_set_id=eval_set_id,
        eval_case_results=eval_case_results,
    )
    # Simulates another process completing its own save into the shared
    # history directory while this save is still in flight.
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text('{}', encoding='utf-8')

  monkeypatch.setattr(
      evaluation_module.LocalEvalSetResultsManager,
      'save_eval_set_result',
      racing_save,
  )
  manager = evaluation_module._ReadableNameEvalSetResultsManager(
      agents_dir=str(tmp_path)
  )

  manager.save_eval_set_result(
      'test_app', 'single.test', [_eval_case_result('case1')]
  )

  assert sorted(p.name for p in history_dir.iterdir()) == [
      'test_app_single.test_1756180000.5.evalset_result.json',
      'test_app_single.test_20260826-123456.evalset_result.json',
  ]


def test_failed_readable_publish_falls_back_to_adk_names(
    tmp_path, monkeypatch
) -> None:
  """When the readable-name path fails, the result must still be published.

  The fallback uses ADK's own unix-timestamp name, whose embedded ids
  already match, so the filename/id invariant holds on the failure path too.
  """

  def _refuse_publish(saved, history_dir):
    raise OSError('simulated hard-link failure')

  monkeypatch.setattr(
      evaluation_module._ReadableNameEvalSetResultsManager,
      '_publish_readable',
      staticmethod(_refuse_publish),
  )
  manager = evaluation_module._ReadableNameEvalSetResultsManager(
      agents_dir=str(tmp_path)
  )

  manager.save_eval_set_result(
      'test_app', 'single.test', [_eval_case_result('case1')]
  )

  history_dir = tmp_path / 'test_app' / '.adk' / 'eval_history'
  files = list(history_dir.iterdir())
  assert len(files) == 1
  saved_file = files[0]
  stem = saved_file.name.removesuffix('.evalset_result.json')
  assert re.fullmatch(r'test_app_single\.test_\d+\.\d+', stem)
  payload = json.loads(saved_file.read_text(encoding='utf-8'))
  assert payload['eval_set_result_id'] == stem
  assert payload['eval_set_result_name'] == stem


@pytest.mark.asyncio
async def test_agent_evaluator_directory_finds_recursive_test_files(
    AgentEvaluator, tmp_path, monkeypatch
) -> None:
  seen_test_files = []
  root_test = tmp_path / 'root.test.json'
  nested = tmp_path / 'nested'
  nested.mkdir()
  nested_test = nested / 'nested.test.json'
  ignored = nested / 'ignored.json'
  for path in [root_test, nested_test, ignored]:
    path.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch, seen_test_files=seen_test_files)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=tmp_path,
      num_runs=1,
  )

  assert set(seen_test_files) == {str(root_test), str(nested_test)}
  assert len(_saved_result_files(AgentEvaluator.results_dir)) == 2


@pytest.mark.asyncio
async def test_agent_evaluator_saves_each_run_for_eval_case(
    AgentEvaluator, tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'multi.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=str(test_file),
      num_runs=3,
  )

  saved_files = _saved_result_files(AgentEvaluator.results_dir)
  saved_result = json.loads(saved_files[0].read_text(encoding='utf-8'))
  assert len(saved_result['eval_case_results']) == 3


@pytest.mark.asyncio
async def test_agent_evaluator_saves_before_raising_for_metric_failure(
    AgentEvaluator, tmp_path, monkeypatch
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
    )

  saved_files = _saved_result_files(AgentEvaluator.results_dir)
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
async def test_agent_evaluator_directory_finds_json_and_toml(
    AgentEvaluator, tmp_path, monkeypatch
) -> None:
  json_test = tmp_path / 'cases.test.json'
  toml_test = tmp_path / 'cases.test.toml'
  ignored_json = tmp_path / 'plain.json'
  ignored_toml = tmp_path / 'plain.toml'
  json_test.write_text('{}', encoding='utf-8')
  toml_test.write_text(_MULTILINE_TOML_EVALSET, encoding='utf-8')
  for path in [ignored_json, ignored_toml]:
    path.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=tmp_path,
      num_runs=1,
  )

  assert len(_saved_result_files(AgentEvaluator.results_dir)) == 2


@pytest.mark.asyncio
async def test_agent_evaluator_directory_skips_non_convention_files(
    AgentEvaluator, tmp_path, monkeypatch
) -> None:
  seen_test_files = []
  test_file = tmp_path / 'cases.test.json'
  # Files that share the .json extension but not the .test. naming convention
  # must be ignored during directory discovery.
  config_file = tmp_path / 'test_config.json'
  result_file = tmp_path / 'cases.evalset_result.json'
  plain_file = tmp_path / 'plain.json'
  for path in [test_file, config_file, result_file, plain_file]:
    path.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch, seen_test_files=seen_test_files)

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=tmp_path,
      num_runs=1,
  )

  assert seen_test_files == [str(test_file)]
  assert len(_saved_result_files(AgentEvaluator.results_dir)) == 1


@pytest.mark.asyncio
async def test_agent_evaluator_direct_convention_file_does_not_warn(
    AgentEvaluator, tmp_path, monkeypatch, caplog
) -> None:
  test_file = tmp_path / 'foo.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  with caplog.at_level('WARNING', logger=evaluation_module.logger.name):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
    )

  assert caplog.records == []
  assert len(_saved_result_files(AgentEvaluator.results_dir)) == 1


@pytest.mark.asyncio
async def test_agent_evaluator_direct_convention_toml_does_not_warn(
    AgentEvaluator, tmp_path, monkeypatch, caplog
) -> None:
  test_file = tmp_path / 'foo.test.toml'
  test_file.write_text(_MULTILINE_TOML_EVALSET, encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  with caplog.at_level('WARNING', logger=evaluation_module.logger.name):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
    )

  assert caplog.records == []
  assert len(_saved_result_files(AgentEvaluator.results_dir)) == 1


@pytest.mark.asyncio
async def test_agent_evaluator_direct_non_convention_json_warns(
    AgentEvaluator, tmp_path, monkeypatch, caplog
) -> None:
  test_file = tmp_path / 'foo.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  with caplog.at_level('WARNING', logger=evaluation_module.logger.name):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
    )

  # The file is still processed despite the non-conventional name.
  assert len(_saved_result_files(AgentEvaluator.results_dir)) == 1
  assert len(caplog.records) == 1
  message = caplog.records[0].getMessage()
  assert str(test_file) in message
  assert 'naming convention' in message


@pytest.mark.asyncio
async def test_agent_evaluator_direct_non_convention_toml_loads_as_toml(
    AgentEvaluator, tmp_path, monkeypatch, caplog
) -> None:
  # A directly specified .toml without the .test. infix should still be parsed
  # by the TOML loader (extension-based routing), with a warning emitted.
  test_file = tmp_path / 'foo.toml'
  test_file.write_text(_MULTILINE_TOML_EVALSET, encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  with caplog.at_level('WARNING', logger=evaluation_module.logger.name):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
    )

  saved_files = _saved_result_files(AgentEvaluator.results_dir)
  assert len(saved_files) == 1
  saved_result = json.loads(saved_files[0].read_text(encoding='utf-8'))
  assert saved_result['eval_set_id'] == 'home_automation'
  assert len(caplog.records) == 1
  assert 'naming convention' in caplog.records[0].getMessage()


@pytest.mark.asyncio
async def test_agent_evaluator_toml_rejects_initial_session_file(
    AgentEvaluator, tmp_path, monkeypatch
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
    )


def test_collect_eval_sets_returns_eval_set_and_config_pairs(
    tmp_path, monkeypatch
) -> None:
  test_file = tmp_path / 'single.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)

  eval_sets = evaluation_module._collect_eval_sets(test_file)

  assert len(eval_sets) == 1
  eval_set, eval_config = eval_sets[0]
  assert eval_set.eval_set_id == 'single.test'
  assert eval_config.user_simulator_config is None


def test_collect_eval_sets_directory_recursive(tmp_path, monkeypatch) -> None:
  seen_test_files = []
  root_test = tmp_path / 'root.test.json'
  nested = tmp_path / 'nested'
  nested.mkdir()
  nested_test = nested / 'nested.test.json'
  ignored = nested / 'ignored.json'
  for path in [root_test, nested_test, ignored]:
    path.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch, seen_test_files=seen_test_files)

  eval_sets = evaluation_module._collect_eval_sets(tmp_path)

  assert set(seen_test_files) == {str(root_test), str(nested_test)}
  assert len(eval_sets) == 2


# --- Custom metrics on the fixture path (issue #11) --------------------------
#
# ADK's ``AgentEvaluator`` builds its ``LocalEvalService`` without a
# ``metric_evaluator_registry``, so unlike ``pytest-adk eval`` this path can
# only register into ADK's process-wide default registry. It therefore has to
# put back what it found: these check both halves of that, and they assert
# from inside the test body, before the conftest safety net's teardown runs.


def _custom_metric_config(
    function_path: str, *, metric_name: str = 'quality'
) -> object:
  from google.adk.evaluation.eval_config import EvalConfig

  return EvalConfig.model_validate({
      'criteria': {metric_name: 0.5},
      'custom_metrics': {
          metric_name: {'code_config': {'name': function_path}}
      },
  })


def _use_config(monkeypatch, eval_config) -> None:
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      'find_config_for_test_file',
      staticmethod(lambda test_file: eval_config),
  )


def _registered_evaluator(metric_name: str):
  """Returns the evaluator class ADK would resolve ``metric_name`` to now."""
  from google.adk.evaluation.metric_evaluator_registry import (
      DEFAULT_METRIC_EVALUATOR_REGISTRY,
  )

  entry = DEFAULT_METRIC_EVALUATOR_REGISTRY._registry.get(metric_name)
  return entry[0] if entry is not None else None


def _capture_evaluator_during_scoring(monkeypatch, metric_name: str) -> list:
  """Records what ``metric_name`` resolves to while ADK is scoring.

  Patched over the stub ``_patch_successful_adk_eval`` installs, because
  ``_get_eval_results_by_eval_id`` is exactly where a real
  ``LocalEvalService`` consults the registry -- so this observes the mapping
  at the only moment it has to be the custom one.
  """
  seen: list = []

  async def get_eval_results_by_eval_id(**kwargs):
    seen.append(_registered_evaluator(metric_name))
    return {'case-1': [_eval_case_result('case-1', 0)]}

  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_eval_results_by_eval_id',
      staticmethod(get_eval_results_by_eval_id),
  )
  return seen


@pytest.mark.asyncio
async def test_fixture_path_registers_config_custom_metrics(
    AgentEvaluator, tmp_path, monkeypatch, write_metric_module
) -> None:
  from google.adk.evaluation.custom_metric_evaluator import (
      _CustomMetricEvaluator,
  )

  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  test_file = tmp_path / 'custom.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  _use_config(monkeypatch, _custom_metric_config(f'{module_name}.sync_metric'))
  scored_with = _capture_evaluator_during_scoring(monkeypatch, 'quality')

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=test_file,
      num_runs=1,
  )

  assert scored_with == [_CustomMetricEvaluator]
  # ...and the process-wide registry is handed back the way it was found.
  assert _registered_evaluator('quality') is None


@pytest.mark.asyncio
async def test_fixture_path_restores_a_shadowed_builtin_evaluator(
    AgentEvaluator, tmp_path, monkeypatch, write_metric_module
) -> None:
  """The leak from the review: a custom metric named after a built-in one.

  Registering ``response_match_score`` as a custom metric used to leave the
  custom evaluator in ADK's process-wide registry, so every *later* test in
  the same pytest session that used the plain built-in metric was scored with
  this test's function instead.
  """
  from google.adk.evaluation.custom_metric_evaluator import (
      _CustomMetricEvaluator,
  )
  from google.adk.evaluation.response_evaluator import ResponseEvaluator

  builtin_before = _registered_evaluator('response_match_score')
  assert builtin_before is ResponseEvaluator  # guards the premise

  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  test_file = tmp_path / 'shadowing.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  _use_config(
      monkeypatch,
      _custom_metric_config(
          f'{module_name}.sync_metric', metric_name='response_match_score'
      ),
  )
  scored_with = _capture_evaluator_during_scoring(
      monkeypatch, 'response_match_score'
  )

  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=test_file,
      num_runs=1,
  )

  # The evalset that asked for the override got it...
  assert scored_with == [_CustomMetricEvaluator]
  # ...and nothing after it does.
  assert _registered_evaluator('response_match_score') is ResponseEvaluator


@pytest.mark.asyncio
async def test_a_later_fixture_eval_uses_the_builtin_evaluator(
    AgentEvaluator, tmp_path, monkeypatch, write_metric_module
) -> None:
  """Two sequential evaluations, standing in for two tests in one session."""
  from google.adk.evaluation.custom_metric_evaluator import (
      _CustomMetricEvaluator,
  )
  from google.adk.evaluation.response_evaluator import ResponseEvaluator

  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  test_file = tmp_path / 'shadowing.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  scored_with = _capture_evaluator_during_scoring(
      monkeypatch, 'response_match_score'
  )

  _use_config(
      monkeypatch,
      _custom_metric_config(
          f'{module_name}.sync_metric', metric_name='response_match_score'
      ),
  )
  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=test_file,
      num_runs=1,
  )

  # A second evalset, whose config names the metric with no custom_metrics
  # entry: it means ADK's built-in, and must be scored with it.
  from google.adk.evaluation.eval_config import EvalConfig

  _use_config(
      monkeypatch,
      EvalConfig.model_validate({'criteria': {'response_match_score': 0.8}}),
  )
  await AgentEvaluator.evaluate(
      agent_module='fake_agent',
      eval_dataset_file_path_or_dir=test_file,
      num_runs=1,
  )

  assert scored_with == [_CustomMetricEvaluator, ResponseEvaluator]


@pytest.mark.asyncio
async def test_fixture_path_restores_the_registry_when_metrics_fail(
    AgentEvaluator, tmp_path, monkeypatch, write_metric_module
) -> None:
  """A failing evalset must not be the one that leaks its metrics."""
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  test_file = tmp_path / 'failing.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  _use_config(monkeypatch, _custom_metric_config(f'{module_name}.sync_metric'))
  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_process_metrics_and_get_failures',
      staticmethod(
          lambda eval_metric_results, print_detailed_results, agent_module: [
              'quality for None Failed. Expected 0.5, but got 0.0.'
          ]
      ),
  )

  with pytest.raises(AssertionError, match='Following are all the test'):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
    )

  assert _registered_evaluator('quality') is None


@pytest.mark.asyncio
async def test_fixture_path_rejects_an_unresolvable_custom_metric(
    AgentEvaluator, tmp_path, monkeypatch
) -> None:
  """The agent must not even be loaded when the config cannot work."""
  test_file = tmp_path / 'custom.test.json'
  test_file.write_text('{}', encoding='utf-8')
  _patch_successful_adk_eval(monkeypatch)
  _use_config(
      monkeypatch,
      _custom_metric_config('no_such_module_for_pytest_adk.metric'),
  )
  loaded_agents = []

  async def get_agent_for_eval(module_name, agent_name=None):
    loaded_agents.append(module_name)
    return object()

  monkeypatch.setattr(
      evaluation_module._AdkAgentEvaluator,
      '_get_agent_for_eval',
      staticmethod(get_agent_for_eval),
  )

  with pytest.raises(ValueError, match='quality'):
    await AgentEvaluator.evaluate(
        agent_module='fake_agent',
        eval_dataset_file_path_or_dir=test_file,
        num_runs=1,
    )

  assert loaded_agents == []
