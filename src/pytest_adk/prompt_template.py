# Copyright 2026 pytest-adk contributors
"""Expand ``<prompt:...>`` template markers in evalset text fields.

An evalset (TOML or JSON) may reference an external prompt file instead of
inlining a (often long) prompt in every eval case. When the entire value of a
``text`` field is a marker like::

    <prompt:prompt.txt VAR1=foo VAR2=ほげ>

the referenced file is read and its placeholders are substituted with the values
from the marker. This lets several eval cases share one common prompt file while
only varying a few variables.

Two rendering engines are supported. The default ``'string'`` engine uses
Python's :class:`string.Template` (``${VAR}`` syntax). The optional ``'jinja'``
engine uses Jinja2 (``{{ VAR }}`` syntax) and requires the ``jinja`` extra
(``pip install "pytest-adk[jinja]"``); it is selected via the
``pytest_adk_prompt_template_engine`` pytest ini option.

The expansion runs after the EvalSet is loaded but before it is handed to ADK's
evaluator, so the agent always sees the fully rendered prompt.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

from google.adk.evaluation.eval_set import EvalSet
from google.genai import types

# The whole (stripped) text value must be a single ``<prompt:...>`` marker; the
# marker is not expanded when embedded inside other text.
_PROMPT_MARKER_RE = re.compile(r'^<prompt:(?P<body>.+)>$', re.DOTALL)

# Supported prompt-template rendering engines. ``'string'`` (the default) keeps
# the historical ``string.Template`` behavior; ``'jinja'`` opts into Jinja2.
_DEFAULT_ENGINE = 'string'
_VALID_ENGINES = ('string', 'jinja')


def _render_string_template(
    template_text: str, variables: dict[str, str], *, filename: str, marker: str
) -> str:
  """Render ``template_text`` with :class:`string.Template` (``${VAR}``)."""
  template = string.Template(template_text)
  try:
    return template.substitute(variables)
  except KeyError as error:
    missing = error.args[0]
    raise ValueError(
        f'Prompt template {filename!r} references variable ${{{missing}}}'
        f' which was not provided in the marker {marker!r}.'
    ) from error


def _render_jinja(
    template_text: str, variables: dict[str, str], *, filename: str, marker: str
) -> str:
  """Render ``template_text`` with Jinja2 (``{{ VAR }}``).

  ``jinja2`` is imported lazily so it stays an optional dependency; it is only
  required when the ``'jinja'`` engine is actually selected. Undefined variables
  raise (via ``StrictUndefined``) to mirror ``string.Template``'s
  error-on-missing behavior, and autoescaping is off because prompts are plain
  text rather than HTML.
  """
  try:
    import jinja2
  except ModuleNotFoundError as error:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "The 'jinja' prompt template engine requires the optional jinja2"
        ' dependency. Install it with: pip install "pytest-adk[jinja]".'
    ) from error

  environment = jinja2.Environment(
      undefined=jinja2.StrictUndefined, autoescape=False
  )
  try:
    return environment.from_string(template_text).render(**variables)
  except jinja2.TemplateError as error:
    raise ValueError(
        f'Failed to render Jinja prompt template {filename!r} referenced by'
        f' marker {marker!r}: {error}.'
    ) from error


def _render_prompt(
    template_text: str,
    variables: dict[str, str],
    engine: str,
    *,
    filename: str,
    marker: str,
) -> str:
  """Render ``template_text`` using the selected ``engine``."""
  if engine == 'jinja':
    return _render_jinja(
        template_text, variables, filename=filename, marker=marker
    )
  return _render_string_template(
      template_text, variables, filename=filename, marker=marker
  )


def _expand_text(
    text: str | None, base_dir: Path, engine: str = _DEFAULT_ENGINE
) -> str | None:
  """Expand ``text`` if it is a ``<prompt:...>`` marker, else return it as-is.

  ``base_dir`` is the directory of the evalset file; the prompt file name in the
  marker is resolved relative to it. ``engine`` selects the rendering engine
  (``'string'`` or ``'jinja'``).
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

  return _render_prompt(
      prompt_path.read_text(encoding='utf-8'),
      variables,
      engine,
      filename=filename,
      marker=text.strip(),
  )


def _expand_content(
    content: types.Content | None, base_dir: Path, engine: str = _DEFAULT_ENGINE
) -> None:
  """Expand prompt markers in every text part of ``content`` in place."""
  if content is None or not getattr(content, 'parts', None):
    return
  for part in content.parts:
    if getattr(part, 'text', None) is not None:
      part.text = _expand_text(part.text, base_dir, engine)


def _expand_prompt_templates(
    eval_set: EvalSet, base_dir: Path, engine: str = _DEFAULT_ENGINE
) -> EvalSet:
  """Expand ``<prompt:...>`` markers in an EvalSet's prompt text fields.

  Walks ``user_content`` and ``final_response`` of every invocation and replaces
  any whole-string ``<prompt:...>`` marker with the rendered prompt file.
  ``engine`` selects the rendering engine (``'string'`` or ``'jinja'``). The
  EvalSet is modified in place and also returned for convenience.
  """
  for eval_case in eval_set.eval_cases:
    # ``conversation`` is None for cases driven by ``conversation_scenario``
    # (the user simulator) instead of static invocations; nothing to expand.
    for invocation in eval_case.conversation or []:
      _expand_content(invocation.user_content, base_dir, engine)
      _expand_content(invocation.final_response, base_dir, engine)
  return eval_set
