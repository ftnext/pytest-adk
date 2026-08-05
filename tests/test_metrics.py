# Copyright 2026 pytest-adk contributors
"""Unit tests for :mod:`pytest_adk.metrics`.

The end-to-end behavior (a configured custom metric actually scoring a run)
lives in ``tests/remote/test_cli.py``; these cover the registration and
validation rules directly, including the ones that are invisible in CLI output
-- such as which ``MetricInfo`` a metric ends up registered under.

``write_metric_module`` (see ``tests/conftest.py``) writes its module into
``tmp_path`` and deliberately leaves ``sys.path`` alone, so tests here make it
importable with ``monkeypatch.syspath_prepend(str(tmp_path))``.
"""

from __future__ import annotations

import pytest
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo

from pytest_adk.metrics import build_metric_evaluator_registry
from pytest_adk.metrics import check_criteria_have_evaluators
from pytest_adk.metrics import check_custom_metrics_are_consistent
from pytest_adk.metrics import custom_metric_info
from pytest_adk.metrics import default_metric_evaluator_registry
from pytest_adk.metrics import register_custom_metrics
from pytest_adk.metrics import registered_custom_metrics
from pytest_adk.metrics import restored_registry_contents


def _config(criteria: dict, custom_metrics: dict | None = None) -> EvalConfig:
  payload: dict = {'criteria': criteria}
  if custom_metrics is not None:
    payload['custom_metrics'] = custom_metrics
  return EvalConfig.model_validate(payload)


def _custom(function_path: str, **extra) -> dict:
  return {'code_config': {'name': function_path}, **extra}


def _registered_names(registry) -> set[str]:
  return {
      metric_info.metric_name
      for metric_info in registry.get_registered_metrics()
  }


def _registered_info(registry, metric_name: str) -> MetricInfo:
  for metric_info in registry.get_registered_metrics():
    if metric_info.metric_name == metric_name:
      return metric_info
  raise AssertionError(f'{metric_name!r} is not registered')


def test_registry_keeps_builtin_metrics_and_adds_the_custom_one(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))

  registry = build_metric_evaluator_registry(
      _config(
          {'tool_trajectory_avg_score': 1.0, 'quality': 0.5},
          {'quality': _custom(f'{module_name}.sync_metric')},
      )
  )

  names = _registered_names(registry)
  assert 'quality' in names
  # Acceptance criterion: built-ins keep their own evaluators.
  assert {'tool_trajectory_avg_score', 'response_match_score'} <= names


def test_default_metric_info_uses_description_and_unit_interval() -> None:
  config = _config(
      {}, {'quality': _custom('m.f', description='How good.')}
  ).custom_metrics['quality']

  metric_info = custom_metric_info('quality', config)

  assert metric_info.metric_name == 'quality'
  assert metric_info.description == 'How good.'
  assert metric_info.metric_value_info.interval == Interval(
      min_value=0.0, max_value=1.0
  )


def test_configured_metric_info_is_respected_and_renamed() -> None:
  config = _config(
      {},
      {
          'quality': _custom(
              'm.f',
              metric_info={
                  # Deliberately disagrees with the custom_metrics key: the
                  # registry keys on metric_name, so the key has to win or the
                  # evaluator lands under a name nothing ever looks up.
                  'metric_name': 'stale_name',
                  'description': 'Bounded at ten.',
                  'metric_value_info': {
                      'interval': {'min_value': -10.0, 'max_value': 10.0}
                  },
              },
          )
      },
  ).custom_metrics['quality']

  metric_info = custom_metric_info('quality', config)

  assert metric_info.metric_name == 'quality'
  assert metric_info.description == 'Bounded at ten.'
  assert metric_info.metric_value_info.interval == Interval(
      min_value=-10.0, max_value=10.0
  )
  # The caller's EvalConfig is left alone (a deep copy was registered).
  assert config.metric_info.metric_name == 'stale_name'


def test_registered_metric_info_matches_the_configured_one(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))

  registry = build_metric_evaluator_registry(
      _config(
          {'quality': 0.5},
          {
              'quality': _custom(
                  f'{module_name}.sync_metric',
                  metric_info={
                      'metric_name': 'quality',
                      'description': 'Bounded at ten.',
                      'metric_value_info': {
                          'interval': {'min_value': -10.0, 'max_value': 10.0}
                      },
                  },
              )
          },
      )
  )

  assert _registered_info(registry, 'quality').metric_value_info == (
      MetricValueInfo(interval=Interval(min_value=-10.0, max_value=10.0))
  )


@pytest.mark.parametrize(
    'function_path_suffix, expected_fragment',
    [
        ('.nope', "has no attribute 'nope'"),
        ('.not_a_function', 'is not callable'),
    ],
)
def test_unresolvable_attribute_is_reported(
    write_metric_module,
    monkeypatch,
    tmp_path,
    function_path_suffix,
    expected_fragment,
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  config = _config(
      {'quality': 0.5},
      {'quality': _custom(module_name + function_path_suffix)},
  )

  with pytest.raises(ValueError) as excinfo:
    build_metric_evaluator_registry(config)

  assert expected_fragment in str(excinfo.value)
  assert 'quality' in str(excinfo.value)


def test_unimportable_module_is_reported() -> None:
  config = _config(
      {'quality': 0.5},
      {'quality': _custom('no_such_module_for_pytest_adk.metric')},
  )

  with pytest.raises(ValueError) as excinfo:
    build_metric_evaluator_registry(config)

  assert 'ModuleNotFoundError' in str(excinfo.value)
  assert '--pythonpath' in str(excinfo.value)


def test_a_metric_module_that_raises_on_import_is_reported(
    monkeypatch, tmp_path
) -> None:
  """Not every import failure is a missing module; a broken one counts too."""
  (tmp_path / 'exploding_metrics.py').write_text(
      "raise RuntimeError('boom')\n", encoding='utf-8'
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  monkeypatch.delitem(
      __import__('sys').modules, 'exploding_metrics', raising=False
  )

  with pytest.raises(ValueError) as excinfo:
    build_metric_evaluator_registry(
        _config(
            {'quality': 0.5}, {'quality': _custom('exploding_metrics.metric')}
        )
    )

  assert 'RuntimeError' in str(excinfo.value)
  assert 'boom' in str(excinfo.value)


def test_function_path_without_a_module_is_reported() -> None:
  config = _config({'quality': 0.5}, {'quality': _custom('metric')})

  with pytest.raises(ValueError, match='fully qualified'):
    build_metric_evaluator_registry(config)


def test_every_bad_metric_is_reported_at_once() -> None:
  config = _config(
      {'a': 0.5, 'b': 0.5},
      {
          'a': _custom('no_such_module_for_pytest_adk.metric'),
          'b': _custom('another_missing_module_for_pytest_adk.metric'),
      },
  )

  with pytest.raises(ValueError) as excinfo:
    build_metric_evaluator_registry(config)

  # One run, one fix-up round: reporting only the first would make a config
  # with two typos take two attempts to correct.
  assert "'a'" in str(excinfo.value)
  assert "'b'" in str(excinfo.value)


def test_nothing_is_registered_when_one_metric_is_unresolvable(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  """A rejected config must not leave half its metrics in the registry."""
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = build_metric_evaluator_registry()

  with pytest.raises(ValueError):
    register_custom_metrics(
        registry,
        _config(
            {'good': 0.5, 'bad': 0.5},
            {
                'good': _custom(f'{module_name}.sync_metric'),
                'bad': _custom('no_such_module_for_pytest_adk.metric'),
            },
        ),
    )

  assert 'good' not in _registered_names(registry)


def test_criteria_metric_without_an_evaluator_is_rejected() -> None:
  registry = build_metric_evaluator_registry()

  with pytest.raises(ValueError) as excinfo:
    check_criteria_have_evaluators(
        registry, _config({'mystery_metric': 0.5}), eval_set_id='some_set'
    )

  assert 'mystery_metric' in str(excinfo.value)
  assert 'some_set' in str(excinfo.value)


def test_criteria_of_builtin_metrics_pass_the_check() -> None:
  registry = build_metric_evaluator_registry()

  check_criteria_have_evaluators(
      registry,
      _config({'tool_trajectory_avg_score': 1.0, 'response_match_score': 0.8}),
      eval_set_id='some_set',
  )


def test_several_configs_all_contribute_their_custom_metrics(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))

  registry = build_metric_evaluator_registry(
      _config({'a': 0.5}, {'a': _custom(f'{module_name}.sync_metric')}),
      _config({'b': 0.5}, {'b': _custom(f'{module_name}.async_metric')}),
  )

  assert {'a', 'b'} <= _registered_names(registry)


def test_identical_definitions_across_configs_are_allowed() -> None:
  check_custom_metrics_are_consistent([
      ('set_one', _config({'quality': 0.5}, {'quality': _custom('m.f')})),
      ('set_two', _config({'quality': 0.9}, {'quality': _custom('m.f')})),
  ])


def test_same_name_pointing_at_different_functions_is_rejected() -> None:
  with pytest.raises(ValueError) as excinfo:
    check_custom_metrics_are_consistent([
        ('set_one', _config({'quality': 0.5}, {'quality': _custom('m.one')})),
        ('set_two', _config({'quality': 0.5}, {'quality': _custom('m.two')})),
    ])

  assert 'set_one' in str(excinfo.value)
  assert 'set_two' in str(excinfo.value)
  assert 'quality' in str(excinfo.value)


def test_same_name_with_different_metric_info_is_rejected() -> None:
  with pytest.raises(ValueError, match='quality'):
    check_custom_metrics_are_consistent([
        ('set_one', _config({'quality': 0.5}, {'quality': _custom('m.f')})),
        (
            'set_two',
            _config(
                {'quality': 0.5},
                {
                    'quality': _custom(
                        'm.f',
                        metric_info={
                            'metric_name': 'quality',
                            'description': 'Different.',
                            'metric_value_info': {
                                'interval': {
                                    'min_value': 0.0,
                                    'max_value': 5.0,
                                }
                            },
                        },
                    )
                },
            ),
        ),
    ])


def test_shadowing_a_builtin_another_evalset_relies_on_is_rejected() -> None:
  with pytest.raises(ValueError) as excinfo:
    check_custom_metrics_are_consistent([
        (
            'set_one',
            _config(
                {'response_match_score': 0.5},
                {'response_match_score': _custom('m.f')},
            ),
        ),
        ('set_two', _config({'response_match_score': 0.8})),
    ])

  assert 'set_two' in str(excinfo.value)
  assert 'response_match_score' in str(excinfo.value)


def test_a_custom_metric_no_other_evalset_uses_is_allowed() -> None:
  check_custom_metrics_are_consistent([
      ('set_one', _config({'quality': 0.5}, {'quality': _custom('m.f')})),
      ('set_two', _config({'response_match_score': 0.8})),
  ])


def test_register_custom_metrics_is_a_no_op_without_custom_metrics() -> None:
  registry = build_metric_evaluator_registry()
  before = _registered_names(registry)

  register_custom_metrics(registry, _config({'response_match_score': 0.8}))

  assert _registered_names(registry) == before


# --- Scoping registrations to one evaluation ---------------------------------


def test_metric_evaluator_registry_still_exposes_its_backing_mapping() -> None:
  """Guard on the one google-adk internal ``restored_registry_contents`` needs.

  ADK has no public way to undo a registration, so the restore reaches for
  ``MetricEvaluatorRegistry._registry``. If a future release renames or drops
  it, fail here with an explanation rather than in the middle of somebody's
  evaluation.
  """
  from google.adk.evaluation.metric_evaluator_registry import (
      MetricEvaluatorRegistry,
  )

  registry = MetricEvaluatorRegistry()
  assert isinstance(getattr(registry, '_registry', None), dict), (
      'MetricEvaluatorRegistry no longer keeps its metric mapping in a'
      ' `_registry` dict; pytest_adk.metrics.restored_registry_contents needs'
      ' another way to snapshot and restore it.'
  )


def test_restored_registry_contents_undoes_a_registration(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = build_metric_evaluator_registry()
  before = _registered_names(registry)

  with restored_registry_contents(registry):
    register_custom_metrics(
        registry,
        _config(
            {'quality': 0.5},
            {'quality': _custom(f'{module_name}.sync_metric')},
        ),
    )
    assert 'quality' in _registered_names(registry)

  assert _registered_names(registry) == before


def test_restored_registry_contents_restores_a_shadowed_builtin(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  from google.adk.evaluation.custom_metric_evaluator import (
      _CustomMetricEvaluator,
  )
  from google.adk.evaluation.response_evaluator import ResponseEvaluator

  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = build_metric_evaluator_registry()

  with restored_registry_contents(registry):
    register_custom_metrics(
        registry,
        _config(
            {'response_match_score': 0.5},
            {
                'response_match_score': _custom(
                    f'{module_name}.sync_metric'
                )
            },
        ),
    )
    assert registry._registry['response_match_score'][0] is (
        _CustomMetricEvaluator
    )

  assert registry._registry['response_match_score'][0] is ResponseEvaluator


def test_restored_registry_contents_restores_when_the_block_raises(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = build_metric_evaluator_registry()
  before = _registered_names(registry)

  with pytest.raises(RuntimeError):
    with restored_registry_contents(registry):
      register_custom_metrics(
          registry,
          _config(
              {'quality': 0.5},
              {'quality': _custom(f'{module_name}.sync_metric')},
          ),
      )
      raise RuntimeError('scoring blew up')

  assert _registered_names(registry) == before


def test_restoration_is_visible_through_every_registry_instance(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  """ADK shares one mapping across instances, so the undo has to as well."""
  from google.adk.evaluation.metric_evaluator_registry import (
      MetricEvaluatorRegistry,
  )

  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = build_metric_evaluator_registry()

  with restored_registry_contents(registry):
    register_custom_metrics(
        registry,
        _config(
            {'quality': 0.5},
            {'quality': _custom(f'{module_name}.sync_metric')},
        ),
    )

  assert 'quality' not in _registered_names(MetricEvaluatorRegistry())


def test_registered_custom_metrics_scopes_the_registration(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = default_metric_evaluator_registry()

  with registered_custom_metrics(
      _config(
          {'quality': 0.5}, {'quality': _custom(f'{module_name}.sync_metric')}
      ),
      eval_set_id='some_set',
  ):
    assert 'quality' in _registered_names(registry)

  assert 'quality' not in _registered_names(registry)


def test_registered_custom_metrics_without_custom_metrics_is_a_no_op() -> None:
  registry = default_metric_evaluator_registry()
  before = _registered_names(registry)

  with registered_custom_metrics(
      _config({'response_match_score': 0.8}), eval_set_id='some_set'
  ):
    assert _registered_names(registry) == before

  assert _registered_names(registry) == before


def test_registered_custom_metrics_leaves_nothing_behind_when_it_rejects(
    write_metric_module, monkeypatch, tmp_path
) -> None:
  """The criteria check runs inside the scope, so its failure must undo too."""
  module_name = write_metric_module()
  monkeypatch.syspath_prepend(str(tmp_path))
  registry = default_metric_evaluator_registry()
  config = _config(
      # 'mystery' has no evaluator, so check_criteria_have_evaluators rejects
      # the config -- after 'quality' has already been registered.
      {'quality': 0.5, 'mystery': 0.5},
      {'quality': _custom(f'{module_name}.sync_metric')},
  )

  with pytest.raises(ValueError, match='mystery'):
    with registered_custom_metrics(config, eval_set_id='some_set'):
      raise AssertionError('the block must not run')

  assert 'quality' not in _registered_names(registry)
