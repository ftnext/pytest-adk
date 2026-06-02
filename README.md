# pytest-adk

## Usage

`AgentEvaluator` is a pytest fixture, auto-registered via the `pytest11` entry
point — installing `pytest-adk` makes it available with no import and no
`conftest.py`. Just request it as a test argument:

```python
import pytest


@pytest.mark.asyncio
async def test_home_automation(AgentEvaluator):
    await AgentEvaluator.evaluate(
        agent_module='home_automation_agent',
        eval_dataset_file_path_or_dir=(
            'tests/integration/fixture/home_automation_agent/'
            'simple_test.test.json'
        ),
    )
```

The fixture binds the eval results directory to pytest's `tmp_path`, so you no
longer pass `results_dir` yourself. Result JSON files are written under
`tmp_path/test_app/.adk/eval_history/`.

## Evalset files: JSON or TOML

`AgentEvaluator.evaluate` discovers and loads evalset files in two formats:

- `*.test.json` — the schema used by google-adk's `AgentEvaluator`.
- `*.test.toml` — the same `EvalSet` schema, written in TOML.

How `eval_dataset_file_path_or_dir` is interpreted depends on whether it points
at a directory or a single file:

- **Directory**: only files matching the `*.test.json` / `*.test.toml` naming
  convention are discovered, recursively. The `.test.` infix is required, so
  sibling files such as `test_config.json` (eval metrics) and the
  `*.evalset_result.json` files written by this helper are naturally excluded —
  no special-casing needed. A plain `data.json` without `.test.` is **not**
  picked up.
- **Single file**: any `.json` or `.toml` file is accepted, since pointing at a
  file is an explicit choice. If the path does not contain `.test.`, a
  `logging.warning` is emitted (under the `pytest_adk.evaluation` logger) noting
  that it falls outside the naming convention, and the file is loaded anyway.
  The loader is chosen by extension: `.toml` → TOML, otherwise JSON.

TOML is handy when a user prompt spans multiple lines: TOML multi-line strings
(`"""..."""`) keep newlines readable, instead of JSON's `\n`-escaped one-liners.
Like JSON, TOML is parsed with the standard library (`tomllib`, Python 3.11+; on
Python 3.10 the [`tomli`](https://pypi.org/project/tomli/) backport is installed
automatically as a dependency).

A `*.test.toml` evalset follows the same `EvalSet` schema as JSON:

```toml
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
```

Notes:

- TOML evalsets support the current `EvalSet` schema only; the legacy data
  format and a separate `initial_session` file (both JSON-only in google-adk)
  are not handled. Express the initial session inside the `EvalSet` instead.
- The companion `test_config.json` (eval metrics / criteria) is unchanged; only
  the evalset data file gains TOML support.

## Prompt templates

When several eval cases share the same (often long) prompt, you can keep the
prompt in a separate file and reference it from a `text` field. If the **entire**
value of a `text` field is a `<prompt:...>` marker, `AgentEvaluator.evaluate`
reads the referenced file, substitutes its variables, and replaces the marker
with the rendered prompt *before* the evalset reaches the evaluator.

Marker syntax:

```
<prompt:FILENAME [KEY=VALUE ...]>
```

Given `prompt.txt`:

```text
Please turn on the ${ROOM} light.
Then confirm it is ${STATE}.
```

an evalset can reference it like this:

```toml
[eval_cases.conversation.user_content]
role = "user"
parts = [ { text = "<prompt:prompt.txt ROOM=living STATE=on>" } ]
```

After expansion the agent sees the fully rendered prompt. This works for both
`*.test.toml` and `*.test.json` evalsets, and applies to both `user_content` and
`final_response` text parts.

Details:

- **Variables** use `string.Template` syntax: `${VAR}` (or `$VAR`).
- `FILENAME` is resolved **relative to the evalset file's directory**.
- The marker must be the **whole** `text` value (leading/trailing whitespace is
  ignored); markers embedded inside other text are not expanded.
- `KEY=VALUE` pairs are **space-separated**, so values cannot contain spaces.
- It is an **error** if the prompt file is missing, a `KEY=VALUE` pair is
  malformed, or the prompt references a variable that the marker does not
  provide.
