# Copyright 2026 pytest-adk contributors

from __future__ import annotations

import pytest
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types

import pytest_adk.evaluation as evaluation_module
from pytest_adk.prompt_template import _expand_prompt_templates


def _eval_set_with_text(user_text: str, final_text: str | None = None) -> EvalSet:
  invocation = Invocation(
      invocationId='inv-1',
      userContent=types.Content(
          role='user', parts=[types.Part(text=user_text)]
      ),
      finalResponse=(
          types.Content(role='model', parts=[types.Part(text=final_text)])
          if final_text is not None
          else None
      ),
  )
  return EvalSet(
      eval_set_id='templated',
      eval_cases=[EvalCase(eval_id='case-1', conversation=[invocation])],
  )


def test_expands_user_content_marker(tmp_path) -> None:
  (tmp_path / 'prompt.txt').write_text(
      'Turn on ${VAR1}. Then say ${VAR2}.', encoding='utf-8'
  )
  eval_set = _eval_set_with_text('<prompt:prompt.txt VAR1=foo VAR2=ほげ>')

  _expand_prompt_templates(eval_set, tmp_path)

  text = eval_set.eval_cases[0].conversation[0].user_content.parts[0].text
  assert text == 'Turn on foo. Then say ほげ.'


def test_marker_is_stripped_before_matching(tmp_path) -> None:
  (tmp_path / 'prompt.txt').write_text('Hello ${VAR1}', encoding='utf-8')
  eval_set = _eval_set_with_text('  <prompt:prompt.txt VAR1=world>\n')

  _expand_prompt_templates(eval_set, tmp_path)

  text = eval_set.eval_cases[0].conversation[0].user_content.parts[0].text
  assert text == 'Hello world'


def test_non_marker_text_is_unchanged(tmp_path) -> None:
  eval_set = _eval_set_with_text('Just a normal prompt with no marker.')

  _expand_prompt_templates(eval_set, tmp_path)

  text = eval_set.eval_cases[0].conversation[0].user_content.parts[0].text
  assert text == 'Just a normal prompt with no marker.'


def test_final_response_is_also_expanded(tmp_path) -> None:
  (tmp_path / 'prompt.txt').write_text('User ${V}', encoding='utf-8')
  (tmp_path / 'expected.txt').write_text('Expected ${V}', encoding='utf-8')
  eval_set = _eval_set_with_text(
      '<prompt:prompt.txt V=in>', '<prompt:expected.txt V=out>'
  )

  _expand_prompt_templates(eval_set, tmp_path)

  invocation = eval_set.eval_cases[0].conversation[0]
  assert invocation.user_content.parts[0].text == 'User in'
  assert invocation.final_response.parts[0].text == 'Expected out'


def test_undefined_variable_raises(tmp_path) -> None:
  (tmp_path / 'prompt.txt').write_text('${VAR1} and ${VAR3}', encoding='utf-8')
  eval_set = _eval_set_with_text('<prompt:prompt.txt VAR1=foo>')

  with pytest.raises(ValueError, match='VAR3'):
    _expand_prompt_templates(eval_set, tmp_path)


def test_missing_prompt_file_raises(tmp_path) -> None:
  eval_set = _eval_set_with_text('<prompt:nope.txt VAR1=foo>')

  with pytest.raises(FileNotFoundError, match='nope.txt'):
    _expand_prompt_templates(eval_set, tmp_path)


def test_invalid_assignment_raises(tmp_path) -> None:
  (tmp_path / 'prompt.txt').write_text('${VAR1}', encoding='utf-8')
  eval_set = _eval_set_with_text('<prompt:prompt.txt VAR1>')

  with pytest.raises(ValueError, match='KEY=VALUE'):
    _expand_prompt_templates(eval_set, tmp_path)


def test_case_without_static_conversation_is_skipped(tmp_path) -> None:
  # Cases driven by conversation_scenario have conversation=None; expansion
  # must not crash trying to iterate it.
  eval_set = EvalSet(
      eval_set_id='scenario',
      eval_cases=[
          EvalCase(
              eval_id='case-1',
              conversation_scenario=ConversationScenario(
                  starting_prompt='Hello', conversation_plan='Chat briefly.'
              ),
          )
      ],
  )

  assert eval_set.eval_cases[0].conversation is None
  _expand_prompt_templates(eval_set, tmp_path)
  assert eval_set.eval_cases[0].conversation is None


_TEMPLATED_TOML_EVALSET = '''\
eval_set_id = "home_automation"

[[eval_cases]]
eval_id = "turn_on_living_room"

[[eval_cases.conversation]]
invocation_id = "inv-1"

[eval_cases.conversation.user_content]
role = "user"
parts = [ { text = "<prompt:prompt.txt ROOM=living>" } ]
'''


def test_toml_evalset_is_expanded_via_loader(tmp_path) -> None:
  (tmp_path / 'prompt.txt').write_text(
      'Please turn on the ${ROOM} light.', encoding='utf-8'
  )
  test_file = tmp_path / 'cases.test.toml'
  test_file.write_text(_TEMPLATED_TOML_EVALSET, encoding='utf-8')

  eval_set = evaluation_module._load_eval_set_from_toml(test_file)
  _expand_prompt_templates(eval_set, test_file.parent)

  text = eval_set.eval_cases[0].conversation[0].user_content.parts[0].text
  assert text == 'Please turn on the living light.'
