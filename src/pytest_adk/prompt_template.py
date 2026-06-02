# Copyright 2026 pytest-adk contributors
"""Expand ``<prompt:...>`` template markers in evalset text fields.

An evalset (TOML or JSON) may reference an external prompt file instead of
inlining a (often long) prompt in every eval case. When the entire value of a
``text`` field is a marker like::

    <prompt:prompt.txt VAR1=foo VAR2=ほげ>

the referenced file is read and its ``${VAR}`` placeholders are substituted with
the values from the marker. This lets several eval cases share one common prompt
file while only varying a few variables.

The expansion runs after the EvalSet is loaded but before it is handed to ADK's
evaluator, so the agent always sees the fully rendered prompt.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

from google.adk.evaluation.eval_set import EvalSet

# The whole (stripped) text value must be a single ``<prompt:...>`` marker; the
# marker is not expanded when embedded inside other text.
_PROMPT_MARKER_RE = re.compile(r'^<prompt:(?P<body>.+)>$', re.DOTALL)


def _expand_text(text: str | None, base_dir: Path) -> str | None:
  """Expand ``text`` if it is a ``<prompt:...>`` marker, else return it as-is.

  ``base_dir`` is the directory of the evalset file; the prompt file name in the
  marker is resolved relative to it.
  """
  if text is None:
    return text
  match = _PROMPT_MARKER_RE.match(text.strip())
  if match is None:
    return text

  tokens = match.group('body').split()
  if not tokens:
    raise ValueError(
        'Prompt template marker is missing a file name: '
        f'{text.strip()!r}.'
    )
  filename, *assignments = tokens

  variables: dict[str, str] = {}
  for assignment in assignments:
    key, sep, value = assignment.partition('=')
    if not sep or not key:
      raise ValueError(
          f'Invalid variable assignment {assignment!r} in prompt template'
          f' marker {text.strip()!r}; expected KEY=VALUE.'
      )
    variables[key] = value

  prompt_path = base_dir / filename
  if not prompt_path.is_file():
    raise FileNotFoundError(
        f'Prompt template file not found: {prompt_path} (referenced by'
        f' {text.strip()!r}).'
    )

  template = string.Template(prompt_path.read_text(encoding='utf-8'))
  try:
    return template.substitute(variables)
  except KeyError as error:
    missing = error.args[0]
    raise ValueError(
        f'Prompt template {filename!r} references variable ${{{missing}}}'
        ' which was not provided in the marker'
        f' {text.strip()!r}.'
    ) from error


def _expand_content(content, base_dir: Path) -> None:
  """Expand prompt markers in every text part of ``content`` in place."""
  if content is None or not getattr(content, 'parts', None):
    return
  for part in content.parts:
    if getattr(part, 'text', None) is not None:
      part.text = _expand_text(part.text, base_dir)


def _expand_prompt_templates(eval_set: EvalSet, base_dir: Path) -> EvalSet:
  """Expand ``<prompt:...>`` markers in an EvalSet's prompt text fields.

  Walks ``user_content`` and ``final_response`` of every invocation and replaces
  any whole-string ``<prompt:...>`` marker with the rendered prompt file. The
  EvalSet is modified in place and also returned for convenience.
  """
  for eval_case in eval_set.eval_cases:
    for invocation in eval_case.conversation:
      _expand_content(invocation.user_content, base_dir)
      _expand_content(invocation.final_response, base_dir)
  return eval_set
