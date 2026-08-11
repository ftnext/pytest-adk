"""Project-local custom metrics for the remote evaluation example."""
# https://gist.github.com/ftnext/9b8ece49af4478b87929f0eff9a2eb5d

from google.adk.evaluation.eval_case import ConversationScenario, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import (
    EvalStatus,
    EvaluationResult,
    PerInvocationResult,
)
from google.adk.evaluation.trajectory_evaluator import get_all_tool_calls


def args_any_support_tool_trajectory_metric(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None = None,
) -> EvaluationResult:
    if expected_invocations is None:
        raise ValueError("expected_invocations is needed by this metric.")
    del conversation_scenario  # unused for per-invocation evaluation

    total_tool_use_accuracy = 0.0
    num_invocations = 0
    per_invocation_results = []

    for actual, expected in zip(actual_invocations, expected_invocations):
        tool_use_accuracy = (
            1.0 if _tool_calls_exact_match_any(actual, expected) else 0.0
        )
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=tool_use_accuracy,
                eval_status=_get_eval_status(tool_use_accuracy),
            )
        )
        total_tool_use_accuracy += tool_use_accuracy
        num_invocations += 1

    if per_invocation_results:
        overall_score = total_tool_use_accuracy / num_invocations
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=_get_eval_status(overall_score),
            per_invocation_results=per_invocation_results,
        )

    return EvaluationResult()


def _tool_calls_exact_match_any(
    actual_invocation: Invocation, expected_invocation: Invocation
) -> bool:
    actual_tool_uses = get_all_tool_calls(actual_invocation.intermediate_data)
    expected_tool_uses = get_all_tool_calls(expected_invocation.intermediate_data)

    if len(actual_tool_uses) != len(expected_tool_uses):
        return False

    for actual, expected in zip(actual_tool_uses, expected_tool_uses):
        if actual.name != expected.name:
            return False
        if not _args_match_any(actual.args, expected.args):
            return False

    return True


def _args_match_any(actual_args, expected_args) -> bool:
    if expected_args == "ANY":
        return True

    if isinstance(expected_args, dict):
        if not isinstance(actual_args, dict):
            return False
        if set(actual_args.keys()) != set(expected_args.keys()):
            return False
        for key, expected_value in expected_args.items():
            if not _args_match_any(actual_args.get(key), expected_value):
                return False
        return True

    if isinstance(expected_args, list):
        if not isinstance(actual_args, list):
            return False
        if len(actual_args) != len(expected_args):
            return False
        return all(
            _args_match_any(actual, expected)
            for actual, expected in zip(actual_args, expected_args)
        )

    return actual_args == expected_args


def _get_eval_status(score: float) -> EvalStatus:
    return EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED
