# pytest-adk

## Evalset files: JSON or TOML

`AgentEvaluator.evaluate` discovers and loads evalset files in two formats:

- `*.json` — the schema used by google-adk's `AgentEvaluator`.
- `*.toml` — the same `EvalSet` schema, written in TOML.

When a directory is passed, every `.json` and `.toml` file under it is loaded
(except the companion `test_config.json`, which holds eval metrics/criteria);
the loader is chosen by extension (`.toml` → TOML, otherwise JSON). The
recommended convention is still to name evalset files `*.test.json` /
`*.test.toml` (this is what google-adk's `AgentEvaluator` looks for). A file
whose path does not contain `.test.` is still processed, but a warning is
logged so accidental matches are easy to spot.

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
