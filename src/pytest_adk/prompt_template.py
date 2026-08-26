# Copyright 2026 pytest-adk contributors
"""Expand ``<prompt:...>`` template markers in evalset text fields.

An evalset (TOML or JSON) may reference an external prompt file instead of
inlining a (often long) prompt in every eval case. When the entire value of a
``text`` field is a marker like::

    <prompt:prompt.txt VAR1=foo VAR2=ほげ>

the referenced file is read and its placeholders are substituted with the values
from the marker. This lets several eval cases share one common prompt file while
only varying a few variables. A file name or a value that contains whitespace
can be quoted::

    <prompt:"my prompt.txt" ROOM="living room">

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

# Quote characters that group whitespace into a single marker token.
_QUOTE_CHARS = '"\''


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


def _split_marker_body(body: str) -> list[str]:
  r"""Split a marker body into a file name and ``KEY=VALUE`` tokens.

  Tokens are whitespace-separated, and a quote (``"`` or ``'``) groups
  whitespace into one token -- but only in the two positions the documented
  forms use: at the start of a token (a quoted file name) and right after the
  first ``=`` of a pair (a quoted value)::

      "my prompt.txt" ROOM="living room"

  Everywhere else a quote is an ordinary character, so a value such as
  ``AUTHOR=O'Reilly's`` keeps its apostrophes instead of being read as quoting
  syntax. There is no backslash escaping either: ``PATTERN=\d+`` keeps its
  backslash, and ``#`` starts no comment. That keeps every marker that worked
  when the body was simply split on whitespace working unchanged; quoting is
  purely additive.

  This is why :func:`shlex.split` is not used directly: it runs in POSIX mode,
  where a backslash escapes the next character and a quote is syntax wherever
  it appears.

  Args:
      body: The text between ``<prompt:`` and the closing ``>``.

  Returns:
      The tokens, with the delimiting quotes removed.

  Raises:
      ValueError: If a quote that opens a quoted run is never closed.
  """
  tokens: list[str] = []
  index = 0
  length = len(body)

  while index < length:
    if body[index].isspace():
      index += 1
      continue

    chars: list[str] = []
    # Position at which a quote would open a quoted run: the token start,
    # then (once the first ``=`` is consumed) the start of the value.
    quote_opens_at = index
    equals_seen = False

    while index < length and not body[index].isspace():
      char = body[index]
      if char in _QUOTE_CHARS and index == quote_opens_at:
        closing = body.find(char, index + 1)
        if closing == -1:
          raise ValueError(
              f'no closing {char} quotation mark'
          )
        chars.append(body[index + 1 : closing])
        index = closing + 1
        continue
      chars.append(char)
      index += 1
      if char == '=' and not equals_seen:
        equals_seen = True
        quote_opens_at = index

    tokens.append(''.join(chars))

  return tokens


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
  marker = text.strip()
  match = _PROMPT_MARKER_RE.match(marker)
  if match is None:
    return text

  try:
    tokens = _split_marker_body(match.group('body'))
  except ValueError as error:
    raise ValueError(
        f'Failed to parse prompt template marker {marker!r}: {error}.'
    ) from error
  if not tokens:
    raise ValueError(
        f'Prompt template marker is missing a file name: {marker!r}.'
    )
  filename, *assignments = tokens

  variables: dict[str, str] = {}
  for assignment in assignments:
    key, sep, value = assignment.partition('=')
    if not sep or not key:
      raise ValueError(
          f'Invalid variable assignment {assignment!r} in prompt template'
          f' marker {marker!r}; expected KEY=VALUE.'
      )
    variables[key] = value

  prompt_path = base_dir / filename
  if not prompt_path.is_file():
    raise FileNotFoundError(
        f'Prompt template file not found: {prompt_path} (referenced by'
        f' {marker!r}).'
    )

  return _render_prompt(
      prompt_path.read_text(encoding='utf-8'),
      variables,
      engine,
      filename=filename,
      marker=marker,
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
