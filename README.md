# pytest-adk

## Evalset files: JSON or TOML

`AgentEvaluator.evaluate` discovers and loads evalset files in two formats:

- `*.test.json` — the schema used by google-adk's `AgentEvaluator`.
- `*.test.toml` — the same `EvalSet` schema, written in TOML.

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
