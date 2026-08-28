# Evaluate a remote ADK agent with an `ANY`-aware custom metric

This example uses `pytest-adk eval` to evaluate a home automation agent running behind `adk api_server`.
Agent inference happens over HTTP, while the evalset is loaded, scored, and saved locally by pytest-adk.

The project-defined `args_any_support_tool_trajectory_metric` compares the
number, names, order, and arguments of tool calls.
An expected argument value of `"ANY"` matches any actual value, which is useful when an argument must be present but its exact value is intentionally unconstrained.

For example, this expected call requires `set_device_info` with the exact
`device_id` and `status`, but accepts any `location` value:

```toml
tool_uses = [
  { name = "set_device_info", args = { device_id = "device_2", status = "OFF", location = "ANY" } },
]
```

## Prerequisites

This example is a standalone project.
From this directory, install its dependencies:

```bash
cd examples/eval-remote-agent
uv sync
```

The evaluated `home_automation_agent` comes from the
[`adk-evaluation` example in agent-practice](https://github.com/ftnext/agent-practice/tree/c4621e25ce24adcc63a6ac969518d5bb8c1a02e5/adk-evaluation).
Have that project available locally before continuing.

## 1. Start the agent

In the `agent-practice/adk-evaluation` project, set a Gemini API key and start the agent as an ADK-compatible HTTP server:

```bash
GOOGLE_API_KEY="your-api-key" uv run adk api_server .
```

`GEMINI_API_KEY` can be used instead of `GOOGLE_API_KEY`.
Keep this process running while evaluating it.

## 2. Run the remote evaluation

In another terminal, return to this example directory and run:

```bash
cd examples/eval-remote-agent

uv run pytest-adk eval \
  http://127.0.0.1:8000 \
  evals \
  --app-name home_automation_agent \
  --num-runs 1 \
  --print-detailed-results
```

The custom metric is installed as the local
`home_automation_agent_evaluation` package (so no `PYTHONPATH` or
`--pythonpath` option is needed).
pytest-adk imports and registers the function specified by `evals/test_config.json` before contacting the agent.

A successful run includes output similar to:

```text
[tests/fixture/home_automation_agent/simple_test.test.json] args_any_support_tool_trajectory_metric: score=1.0 threshold=1.0 status=PASSED
[turn_off_device_2] args_any_support_tool_trajectory_metric: score=1.0 threshold=1.0 status=PASSED
Eval results saved under: home_automation_agent/.adk/eval_history
```

`--print-detailed-results` is what makes those passing lines appear (without
it, only metrics that did not pass are reported). On a *failing* run the same
flag additionally prints a per-invocation breakdown table on stderr, showing
the prompt next to the expected and actual response and tool calls.

Exact formatting may vary by google-adk version. The command exits with

* `0` when all metrics pass
* `1` when a metric fails
* `2` when inference or the evaluation setup fails

The equivalent JSON evalset is available at `evals/args_any_support.test.json`.
Point the command at either the TOML file or the JSON file.
Passing the whole `evals` directory evaluates both files.

## How the custom metric works

ADK 1.24.0 and later call a custom metric with four arguments:

```python
def args_any_support_tool_trajectory_metric(
    eval_metric,
    actual_invocations,
    expected_invocations,
    conversation_scenario=None,
):
    ...
```

The leading `eval_metric` parameter is required by the newer signature even though this metric does not otherwise use it.
Omitting the parameter uses the ADK 1.23.0 signature and causes a positional-argument `TypeError` with newer ADK versions.

For every invocation, the metric:

1. Extracts tool calls with ADK's `get_all_tool_calls`, which supports both `IntermediateData` eval expectations and event-based remote inference.
2. Requires the actual and expected trajectories to contain the same number of tool calls in the same order.
3. Requires each tool name to match.
4. Recursively compares argument dictionaries and lists, treating an expected value of `"ANY"` as a wildcard.

Argument dictionaries must still contain exactly the same keys.
For example, `location = "ANY"` accepts any location value but does not accept a call that omits the `location` argument.

The evaluation config registers the function by its fully qualified import
path:

```json
{
  "criteria": {
    "args_any_support_tool_trajectory_metric": 1.0
  },
  "custom_metrics": {
    "args_any_support_tool_trajectory_metric": {
      "code_config": {
        "name": "home_automation_agent_evaluation.args_any_support_tool_trajectory_metric"
      }
    }
  }
}
```

Inference happens in the `adk api_server` process.
Evalset loading, custom metric execution, result reporting, and result persistence happen in the `pytest-adk eval` process.
