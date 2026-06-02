# Copyright 2026 pytest-adk contributors
"""Generate a fill-in evalset template from the ADK ``EvalSet`` schema.

Authoring an evalset file from scratch is tedious because of its deep, nested
shape. This module emits a placeholder ``EvalSet`` (one eval case with one
conversation turn) serialized as TOML (default) or JSON, so users can copy it
and replace the ``REPLACE_ME`` placeholders instead of writing the structure by
hand.

The template is built from ADK's own models and serialized, so it always tracks
the current ``EvalSet`` schema rather than a hand-maintained literal.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import tomli_w
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types

_PLACEHOLDER = 'REPLACE_ME'
_FORMATS = ('toml', 'json')


def _placeholder_eval_set() -> EvalSet:
  """Build an EvalSet with placeholder values for one conversation turn."""
  return EvalSet(
      eval_set_id=_PLACEHOLDER,
      eval_cases=[
          EvalCase(
              eval_id=_PLACEHOLDER,
              conversation=[
                  Invocation(
                      invocation_id='inv-1',
                      user_content=types.Content(
                          role='user',
                          parts=[types.Part(text=f'{_PLACEHOLDER}: user prompt')],
                      ),
                      final_response=types.Content(
                          role='model',
                          parts=[
                              types.Part(
                                  text=f'{_PLACEHOLDER}: expected model response'
                              )
                          ],
                      ),
                  )
              ],
          )
      ],
  )


def eval_set_template(fmt: str = 'toml') -> str:
  """Return a fill-in evalset template serialized as ``fmt`` (toml or json).

  Uses snake_case keys (matching the documented examples) and drops ``None`` and
  default-valued fields. Dropping ``None`` is required for TOML, which has no
  null representation.

  Args:
      fmt: ``"toml"`` or ``"json"``. TOML is the default because it is easier
          to edit for multi-line prompts.

  Returns:
      A serialized ADK ``EvalSet`` with ``REPLACE_ME`` placeholders.

  Raises:
      ValueError: If ``fmt`` is not one of the supported output formats.
  """
  if fmt not in _FORMATS:
    raise ValueError(f'Unsupported format {fmt!r}; choose one of {_FORMATS}.')

  data = _placeholder_eval_set().model_dump(
      mode='json',
      by_alias=False,
      exclude_none=True,
      exclude_defaults=True,
  )
  if fmt == 'toml':
    return tomli_w.dumps(data, multiline_strings=True)
  return json.dumps(data, indent=2, ensure_ascii=False) + '\n'


def main(argv: Sequence[str] | None = None) -> None:
  """Print or write a fill-in evalset template for the console script.

  Args:
      argv: Optional command-line arguments. ``None`` means use
          :data:`sys.argv`, via :mod:`argparse`.
  """
  parser = argparse.ArgumentParser(
      prog='pytest-adk-eval-schema',
      description='Generate a fill-in evalset template (EvalSet schema).',
  )
  parser.add_argument(
      '-f',
      '--format',
      choices=_FORMATS,
      default='toml',
      help='Output format (default: toml).',
  )
  parser.add_argument(
      '-o',
      '--output',
      help='Write the template to this path (default: stdout).',
  )
  parser.add_argument(
      '--force',
      action='store_true',
      help='Overwrite the output file if it already exists.',
  )
  args = parser.parse_args(argv)

  content = eval_set_template(args.format)

  if args.output is None:
    print(content, end='')
    return

  output_path = Path(args.output)
  if output_path.exists() and not args.force:
    parser.error(
        f'{output_path} already exists; pass --force to overwrite it.'
    )
  output_path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
  main()
