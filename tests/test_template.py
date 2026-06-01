# Copyright 2026 pytest-adk contributors

from __future__ import annotations

import json

try:
  import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
  import tomli as tomllib

import pytest
from google.adk.evaluation.eval_set import EvalSet

import pytest_adk.evaluation as evaluation_module
from pytest_adk import eval_set_template
from pytest_adk.template import main


def test_toml_template_is_valid_against_schema() -> None:
  template = eval_set_template('toml')

  data = tomllib.loads(template)
  eval_set = EvalSet.model_validate(data)

  assert 'eval_set_id' in data
  assert 'eval_cases' in data
  assert eval_set.eval_set_id == 'REPLACE_ME'
  # TOML cannot represent null, so the template must not contain one.
  assert 'null' not in template


def test_json_template_is_valid_against_schema() -> None:
  template = eval_set_template('json')

  data = json.loads(template)
  eval_set = EvalSet.model_validate(data)

  assert eval_set.eval_set_id == 'REPLACE_ME'
  invocation = eval_set.eval_cases[0].conversation[0]
  assert invocation.user_content.parts[0].text == 'REPLACE_ME: user prompt'


def test_default_format_is_toml() -> None:
  assert eval_set_template() == eval_set_template('toml')


def test_toml_template_loads_through_real_loader(tmp_path) -> None:
  test_file = tmp_path / 'generated.test.toml'
  test_file.write_text(eval_set_template('toml'), encoding='utf-8')

  eval_set = evaluation_module._load_eval_set_from_toml(test_file)

  assert eval_set.eval_set_id == 'REPLACE_ME'


def test_unsupported_format_raises() -> None:
  with pytest.raises(ValueError):
    eval_set_template('yaml')


def test_main_prints_json_to_stdout(capsys) -> None:
  main(['--format', 'json'])

  out = capsys.readouterr().out
  EvalSet.model_validate(json.loads(out))


def test_main_writes_toml_file_by_default(tmp_path) -> None:
  output = tmp_path / 'x.test.toml'

  main(['-o', str(output)])

  assert EvalSet.model_validate(tomllib.loads(output.read_text())).eval_set_id


def test_main_refuses_to_overwrite_without_force(tmp_path) -> None:
  output = tmp_path / 'x.test.toml'
  output.write_text('eval_set_id = "keep"\n', encoding='utf-8')

  with pytest.raises(SystemExit):
    main(['-o', str(output)])

  assert 'keep' in output.read_text()


def test_main_overwrites_with_force(tmp_path) -> None:
  output = tmp_path / 'x.test.toml'
  output.write_text('eval_set_id = "keep"\n', encoding='utf-8')

  main(['-o', str(output), '--force'])

  assert 'REPLACE_ME' in output.read_text()
