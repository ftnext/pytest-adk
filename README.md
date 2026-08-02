# pytest-adk

Pytest helpers for evaluating agents built with
[Google ADK](https://github.com/google/adk-python). The package provides:

- an auto-registered `AgentEvaluator` pytest fixture that saves ADK eval result
  JSON files under each test's `tmp_path`;
- TOML evalset support, including multi-line prompts;
- external prompt templates for repeated evalset text, rendered with
  `string.Template` by default or optionally with Jinja2;
- a `pytest-adk-eval-schema` CLI for generating fill-in evalset templates;
- helpers for resuming an exported ADK session with an in-memory `Runner`.

## Installation

```bash
pip install pytest-adk
```

For development and tests, install the `dev` extra:

```bash
pip install "pytest-adk[dev]"
```

Both google-adk v1 (>=1.30.0) and v2 are supported.

## Running the tests

Both google-adk major versions are covered by [Hatch](https://hatch.pypa.io/)
environments (`test-adk1` and `test-adk2`) declared in `pyproject.toml`. Each
one installs its dependencies with uv:

```bash
hatch run test-adk1:run
hatch run test-adk2:run
```

Without Hatch installed, `uvx hatch run test-adk1:run` works too.

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

After the run, pytest's terminal summary prints an `ADK eval results` section
listing, for every test that used the fixture, the `eval_history` directory
where its results were saved — shown regardless of whether the test passed or
failed, so you can always find them:

```
=================== ADK eval results ===================
tests/test_home_automation.py::test_home_automation
  /tmp/pytest-of-you/pytest-0/test_home_automation0/test_app/.adk/eval_history
```

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

- **Variables** use `string.Template` syntax by default: `${VAR}` (or `$VAR`).
- `FILENAME` is resolved **relative to the evalset file's directory**.
- The marker must be the **whole** `text` value (leading/trailing whitespace is
  ignored); markers embedded inside other text are not expanded.
- `KEY=VALUE` pairs are **space-separated**, so values cannot contain spaces.
- It is an **error** if the prompt file is missing, a `KEY=VALUE` pair is
  malformed, or the prompt references a variable that the marker does not
  provide.

### Jinja prompt templates

By default the prompt file is rendered with `string.Template` (`${VAR}`). To use
Jinja2 (`{{ VAR }}`) syntax instead, install the optional extra and opt in via
the `pytest_adk_prompt_template_engine` ini option in `pyproject.toml`:

```console
pip install "pytest-adk[jinja]"
```

```toml
[tool.pytest.ini_options]
pytest_adk_prompt_template_engine = "jinja"
```

With the Jinja engine selected, the same `prompt.txt` would be written as:

```text
Please turn on the {{ ROOM }} light.
Then confirm it is {{ STATE }}.
```

The marker syntax (`<prompt:FILENAME KEY=VALUE ...>`) is unchanged; only the
placeholder syntax inside the prompt file differs. Referencing a variable that
the marker does not provide is an **error** (Jinja runs with
`StrictUndefined`).

## Generate an evalset template

Use `pytest-adk-eval-schema` to generate a minimal `EvalSet` file with
`REPLACE_ME` placeholders:

```bash
pytest-adk-eval-schema -o tests/evals/example.test.toml
```

TOML is the default output format. JSON is also available:

```bash
pytest-adk-eval-schema --format json
```

The command refuses to overwrite an existing file unless you pass `--force`.
The same generator is available from Python:

```python
from pytest_adk import eval_set_template

template = eval_set_template("toml")
```

## Resume an exported ADK session

`load_session_from_json` reads a session exported by ADK from either a file path
or a raw JSON string. `runner_from_exported_session` restores that session into
an in-memory ADK `Runner`, copying the exported state and replaying events via
the session service.

```python
from pathlib import Path

from google.genai import types
from pytest_adk import runner_from_exported_session
from your_agent.agent import root_agent


async def test_resume_exported_session():
    runner, session = await runner_from_exported_session(
        root_agent,
        Path("tests/fixtures/roll_die.session.json"),
    )

    events = runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What numbers did I get?")],
        ),
    )
    async for _ in events:
        pass
```

You can override `app_name`, `user_id`, or `session_id` when restoring, and you
can pass custom artifact, memory, or credential services. If you do not provide
services, in-memory ADK services are used.

## Evaluating a deployed agent: `pytest-adk eval`

`pytest-adk eval` evaluates an ADK agent that is already running behind an
HTTP endpoint (e.g. deployed to Cloud Run), instead of importing an agent
module and running it in-process. Inference is delegated to the remote
server; evalset loading, scoring, and result persistence reuse the same local
ADK evaluation machinery as the `AgentEvaluator` fixture.

The target must be an `adk api_server`-compatible REST endpoint (the
deployment form ADK-based agents typically take). We recommend the server run
a roughly similar google-adk generation to your local environment: unknown
response fields are ignored, but ADK's own semantic changes across versions
are not otherwise protected against.

```bash
pytest-adk eval https://my-agent.example.com \
  tests/evals/ \
  --app-name my_agent \
  --header 'Authorization: Bearer <token>' \
  --num-runs 3
```

`EVAL_SET_PATH` accepts one or more evalset files or directories, using the
same `.test.json` / `.test.toml` discovery convention as the fixture.
`--app-name` can be omitted when the server's `GET /list-apps` lists exactly
one app. See `pytest-adk eval --help` for the full flag list (`--user-id`,
`--config-file-path`, `--timeout`, `--parallelism`, `--results-dir`,
`--prompt-template-engine`, `--keep-sessions`, `--print-detailed-results`).

`<prompt:...>` markers in the evalsets are rendered the same way as on the
fixture path, but the engine is selected with `--prompt-template-engine`
(`string`, the default, or `jinja`):

```bash
pytest-adk eval https://my-agent.example.com tests/evals/ \
  --prompt-template-engine jinja
```

Results are saved via ADK's standard `LocalEvalSetResultsManager` layout,
under `{results-dir}/{app-name}/.adk/eval_history/` (`--results-dir` defaults
to the current directory). The save path is always printed after the run.

The command exits `0` if every eval metric passed, `1` if at least one metric
failed, and `2` on an execution error (bad `AGENT_URL`, a connection failure,
`--app-name` resolution failure, an evalset or `--config-file-path` that cannot
be loaded, `EVAL_SET_PATH`s that discover no evalset at all, two evalsets
sharing one `eval_set_id`, or any eval case whose inference failed).

Scoring always runs locally: LLM-as-judge metrics use your own API key and
billing, and only inference is delegated to the remote server.

### Dependencies on google-adk v2

pytest-adk ships a minimal subset of google-adk's `eval` extra (`pandas`,
`rouge-score`, `tabulate`) as normal dependencies, which is enough to run
evaluations on google-adk v1. On google-adk v2, ADK's evaluation import chain
additionally requires the `vertexai` module even though remote evaluation
never talks to Vertex AI; install the base `google-cloud-aiplatform` package
(or `google-adk[eval]`) to run `pytest-adk eval` there. Without it, the
command prints a clear error instead of a traceback.

### Limitations

- The prompt-template engine is **not** auto-discovered from the
  `pytest_adk_prompt_template_engine` pytest ini option: the CLI does not load
  pytest config, so a project set to `jinja` must pass
  `--prompt-template-engine jinja` explicitly. Otherwise `string.Template`
  silently leaves `{{ VAR }}` placeholders unrendered and both the deployed
  agent and the local scorer see literal template syntax.
- Eval cases using `conversation_scenario` (the user-simulator, dynamic
  multi-turn form) are not supported and fail with a clear per-case message;
  only static `conversation` eval cases can be run remotely.
- `app_details` (e.g. tool declarations) is not available for remote runs, so
  rubric-style metrics that need it may degrade or not work.
- ADK's eval-internal plugins don't run on the remote server, so remote
  evaluation measures your agent's production configuration as-is, not the
  instrumented local eval path.
- Remote tools with real-world side effects **will** execute, and with the
  default `--parallelism` of 4, potentially concurrently.
- Reusing an existing remote session (instead of creating a fresh one) is
  only possible against google-adk v2 servers, via an extra `session_id`
  field on an eval case's `session_input`.
- Sessions created for the run are deleted afterwards unless you pass
  `--keep-sessions`.
