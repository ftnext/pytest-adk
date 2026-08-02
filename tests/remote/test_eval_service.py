# Copyright 2026 pytest-adk contributors
"""Tests for RemoteEvalService.

Two kinds of coverage live here:

- A guard test (mirrors ROADMAP task 0-2's inspect-based pattern) that keeps
  watch on the one google-adk internal fact RemoteEvalService's dummy-agent
  trick depends on: ``LocalEvalService`` reads ``self._root_agent`` only
  inside ``perform_inference()``.
- End-to-end tests that run RemoteEvalService's ``perform_inference()`` +
  the inherited ``evaluate()`` against a fake ``api_server`` (a small FastAPI
  app wired in-process via ``httpx.ASGITransport``), using only LLM-free
  metrics (``tool_trajectory_avg_score``, ``response_match_score``).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import SessionInput
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.genai import types

from pytest_adk.remote.eval_service import RemoteEvalService

from .fake_server import FakeApiServer
from .fake_server import REMOTE_APP_NAME as _REMOTE_APP_NAME
from .fake_server import client_for as _client_for
from .fake_server import content as _content
from .fake_server import function_call_event as _function_call_event
from .fake_server import text_event as _text_event

_LOCAL_APP_NAME = 'test_app'  # namespace inside InMemoryEvalSetsManager


def test_remote_package_lazily_exports_remote_eval_service() -> None:
  """``pytest_adk.remote.RemoteEvalService`` resolves via ``__getattr__``.

  See ``pytest_adk/remote/__init__.py``'s module docstring: exporting it
  lazily (instead of with a plain top-level import) keeps
  ``import pytest_adk.remote`` -- and therefore ``AdkApiClient`` -- usable
  even where ``RemoteEvalService``'s import chain (``vertexai`` on
  google-adk v2) is unavailable, since ``eval_service.py`` isn't touched
  until ``RemoteEvalService`` is actually accessed.
  """
  import pytest_adk.remote as remote_package

  assert 'RemoteEvalService' not in vars(remote_package)  # not eager
  assert remote_package.RemoteEvalService is RemoteEvalService  # resolves
  with pytest.raises(AttributeError):
    remote_package.does_not_exist  # unknown attributes still raise normally


# ---------------------------------------------------------------------------
# Guard test (ROADMAP task 0-2 pattern)
# ---------------------------------------------------------------------------


def test_local_eval_service_root_agent_referenced_only_in_perform_inference() -> (
    None
):
  """RemoteEvalService's dummy root_agent is only safe while this holds.

  RemoteEvalService (pytest_adk/remote/eval_service.py) passes a dummy
  ``BaseAgent`` to ``LocalEvalService.__init__`` and fully overrides
  ``perform_inference()`` (the only place ``LocalEvalService`` reads
  ``self._root_agent``, per its docstring), relying on ``evaluate()`` and
  everything it calls to never touch ``self._root_agent``. This test scans
  every ``LocalEvalService`` method except ``__init__`` (which merely stores
  the constructor argument) for the literal attribute access
  ``self._root_agent``.

  Note this deliberately checks for ``self._root_agent`` rather than the
  bare substring ``_root_agent``: methods such as
  ``_perform_inference_single_eval_item`` call
  ``EvaluationGenerator._generate_inferences_from_root_agent(...)``, whose
  name also contains the substring ``_root_agent`` but is unrelated to the
  ``self._root_agent`` attribute -- a naive substring search would flag it
  as a false positive.
  """
  offending_methods = []
  for name, member in inspect.getmembers(
      LocalEvalService, predicate=inspect.isfunction
  ):
    if name in ('__init__', 'perform_inference'):
      continue
    source = inspect.getsource(member)
    if 'self._root_agent' in source:
      offending_methods.append(name)

  assert not offending_methods, (
      'LocalEvalService methods other than __init__ now reference'
      f' `self._root_agent`: {offending_methods}. RemoteEvalService'
      " (src/pytest_adk/remote/eval_service.py) relies on this NOT being the"
      ' case for its dummy-root-agent construction to be safe -- see its'
      ' class docstring and REMOTE_EVAL_PLAN.md section 7. If this fails, a'
      ' google-adk upgrade means the dummy agent could now leak into scoring'
      ' and pytest_adk.remote.eval_service must be revisited.'
  )
  # perform_inference() itself must still reference it -- otherwise this
  # guard would be checking a stale assumption for the wrong reason.
  assert 'self._root_agent' in inspect.getsource(
      LocalEvalService.perform_inference
  )


async def _register_and_perform_inference(
    service: RemoteEvalService, eval_set: EvalSet
) -> list[InferenceResult]:
  """Registers ``eval_set`` on the service's manager and runs perform_inference()."""
  eval_sets_manager = service._eval_sets_manager
  eval_sets_manager.create_eval_set(
      app_name=_LOCAL_APP_NAME, eval_set_id=eval_set.eval_set_id
  )
  for eval_case in eval_set.eval_cases:
    eval_sets_manager.add_eval_case(
        app_name=_LOCAL_APP_NAME,
        eval_set_id=eval_set.eval_set_id,
        eval_case=eval_case,
    )

  inference_request = InferenceRequest(
      app_name=_LOCAL_APP_NAME,
      eval_set_id=eval_set.eval_set_id,
      inference_config=InferenceConfig(),
  )
  return [
      result
      async for result in service.perform_inference(inference_request)
  ]


async def _evaluate(
    service: RemoteEvalService,
    inference_results: list[InferenceResult],
    eval_metrics: list[EvalMetric],
) -> list[EvalCaseResult]:
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
  )
  return [result async for result in service.evaluate(evaluate_request)]


async def _run_full_eval(
    service: RemoteEvalService,
    eval_set: EvalSet,
    eval_metrics: list[EvalMetric],
) -> tuple[list[InferenceResult], list[EvalCaseResult]]:
  """Registers ``eval_set`` and drives perform_inference() + evaluate()."""
  inference_results = await _register_and_perform_inference(service, eval_set)
  eval_case_results = await _evaluate(service, inference_results, eval_metrics)
  return inference_results, eval_case_results


def _session_input_supports_extra_session_id() -> bool:
  """Probes whether this google-adk's SessionInput will retain an extra field.

  google-adk v2's ``SessionInput`` sets ``model_config = ConfigDict(extra=
  "allow")``, so ``session_id`` survives even though it isn't a declared
  field. google-adk v1's ``SessionInput`` inherits ``extra="forbid"`` from
  ``EvalBaseModel`` and silently drops unknown fields even via
  ``model_construct()`` (which skips validation but still honors the
  model's extra-field policy for what gets stored). See
  REMOTE_EVAL_PLAN.md sec 9 on version skew.
  """
  probe = SessionInput.model_construct(
      app_name='a', user_id='b', session_id='c'
  )
  return getattr(probe, 'session_id', None) == 'c'


async def _evaluate_tolerating_known_adk_v1_none_inferences_bug(
    service: RemoteEvalService,
    inference_results: list[InferenceResult],
    eval_metrics: list[EvalMetric],
) -> list[EvalCaseResult]:
  """Runs evaluate(), skipping the test on a known pre-v2 google-adk bug.

  google-adk v1's ``LocalEvalService._evaluate_single_inference_result()``
  unconditionally computes ``len(inference_result.inferences)``; google-adk
  v2 added an explicit early return for ``inferences is None`` (see
  ``local_eval_service.py``). Because RemoteEvalService reports a failed eval
  case as an ``InferenceResult`` with ``inferences=None`` -- exactly what
  ``BaseEvalService``'s contract calls for -- mixing a FAILURE result into
  one ``evaluate()`` call can raise ``TypeError`` under pre-v2 google-adk.
  This is an upstream limitation of that ADK version's inherited,
  unmodified ``evaluate()``, not a RemoteEvalService bug (see
  REMOTE_EVAL_PLAN.md sec 9 on version skew), so the caller skips rather
  than fails when it is hit.
  """
  try:
    return await _evaluate(service, inference_results, eval_metrics)
  except TypeError as e:
    pytest.skip(
        "This google-adk version's LocalEvalService.evaluate() raises"
        ' TypeError for a FAILURE InferenceResult with inferences=None (a'
        ' gap only closed in google-adk v2, see REMOTE_EVAL_PLAN.md sec 9);'
        f' not a RemoteEvalService bug. Original error: {e}'
    )
    raise AssertionError('unreachable')  # pytest.skip always raises


def _weather_eval_case(eval_id: str, user_id: str) -> EvalCase:
  return EvalCase(
      eval_id=eval_id,
      session_input=SessionInput(
          app_name=_REMOTE_APP_NAME,
          user_id=user_id,
          state={'locale': 'en-US'},
      ),
      conversation=[
          Invocation(
              user_content=_content('what is the weather in Tokyo?'),
              final_response=types.Content(
                  role='model',
                  parts=[types.Part(text='It is sunny in Tokyo.')],
              ),
              intermediate_data=IntermediateData(
                  tool_uses=[
                      types.FunctionCall(
                          name='get_weather', args={'city': 'Tokyo'}
                      )
                  ]
              ),
          ),
          Invocation(
              user_content=_content('what time is it there?'),
              final_response=types.Content(
                  role='model', parts=[types.Part(text='It is 3pm in Tokyo.')]
              ),
              intermediate_data=IntermediateData(
                  tool_uses=[
                      types.FunctionCall(
                          name='get_time', args={'city': 'Tokyo'}
                      )
                  ]
              ),
          ),
      ],
  )


def _weather_script() -> list[list[dict[str, Any]]]:
  return [
      [
          _function_call_event('', 'get_weather', {'city': 'Tokyo'}),
          _text_event('', 'It is sunny in Tokyo.'),
      ],
      [
          _function_call_event('', 'get_time', {'city': 'Tokyo'}),
          _text_event('', 'It is 3pm in Tokyo.'),
      ],
  ]


_METRICS = [
    EvalMetric(metric_name='tool_trajectory_avg_score', threshold=1.0),
    EvalMetric(metric_name='response_match_score', threshold=0.8),
]


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


async def test_multi_turn_eval_case_scores_as_expected_and_cleans_up_session() -> (
    None
):
  server = FakeApiServer()
  server.scripts['alice'] = _weather_script()
  client = _client_for(server)
  eval_sets_manager = InMemoryEvalSetsManager()
  service = RemoteEvalService(
      client,
      app_name=_REMOTE_APP_NAME,
      eval_sets_manager=eval_sets_manager,
  )
  eval_set = EvalSet(
      eval_set_id='weather_set',
      eval_cases=[_weather_eval_case('weather_case', 'alice')],
  )

  try:
    inference_results, eval_case_results = await _run_full_eval(
        service, eval_set, _METRICS
    )
  finally:
    await client.aclose()

  assert len(inference_results) == 1
  inference_result = inference_results[0]
  assert inference_result.status == InferenceStatus.SUCCESS
  assert inference_result.inferences is not None
  assert len(inference_result.inferences) == 2  # two turns

  assert len(eval_case_results) == 1
  eval_case_result = eval_case_results[0]
  assert eval_case_result.eval_id == 'weather_case'
  assert eval_case_result.final_eval_status == EvalStatus.PASSED
  scores = {
      m.metric_name: m.score for m in eval_case_result.overall_eval_metric_results
  }
  assert scores['tool_trajectory_avg_score'] == pytest.approx(1.0)
  assert scores['response_match_score'] == pytest.approx(1.0)

  # Initial session state from session_input was POSTed to the fake server.
  assert len(server.create_session_requests) == 1
  create_request = server.create_session_requests[0]
  assert create_request['app_name'] == _REMOTE_APP_NAME
  assert create_request['user_id'] == 'alice'
  assert create_request['body']['state'] == {'locale': 'en-US'}

  # Multi-turn conversation accumulated events across turns: both /run calls
  # used the server-assigned session id, not the (never sent back) requested
  # one.
  assert len(server.run_requests) == 2
  assert server.run_requests[0]['sessionId'] == server.run_requests[1]['sessionId']
  server_session_id = server.run_requests[0]['sessionId']
  assert server_session_id == inference_result.session_id
  assert server_session_id.startswith('srv-session-')

  # The session this service created was deleted afterwards.
  assert server.deleted_session_ids == [server_session_id]


async def test_keep_sessions_true_does_not_delete_created_session() -> None:
  server = FakeApiServer()
  server.scripts['bob'] = _weather_script()
  client = _client_for(server)
  eval_sets_manager = InMemoryEvalSetsManager()
  service = RemoteEvalService(
      client,
      app_name=_REMOTE_APP_NAME,
      eval_sets_manager=eval_sets_manager,
      keep_sessions=True,
  )
  eval_set = EvalSet(
      eval_set_id='weather_set',
      eval_cases=[_weather_eval_case('weather_case', 'bob')],
  )

  try:
    inference_results, _ = await _run_full_eval(service, eval_set, _METRICS)
  finally:
    await client.aclose()

  assert inference_results[0].status == InferenceStatus.SUCCESS
  assert server.create_session_requests  # a session WAS created
  assert server.deleted_session_ids == []  # but not deleted


async def test_existing_session_is_reused_and_never_created_or_deleted() -> (
    None
):
  if not _session_input_supports_extra_session_id():
    pytest.skip(
        "This google-adk version's SessionInput does not retain an extra"
        ' session_id field (see REMOTE_EVAL_PLAN.md sec 9 on version skew);'
        ' the existing-session-reuse feature is not usable here.'
    )

  server = FakeApiServer()
  server.scripts['carol'] = _weather_script()
  client = _client_for(server)
  eval_sets_manager = InMemoryEvalSetsManager()
  service = RemoteEvalService(
      client,
      app_name=_REMOTE_APP_NAME,
      eval_sets_manager=eval_sets_manager,
  )
  eval_case = _weather_eval_case('weather_case', 'carol')
  # An explicit session_id means "use this pre-existing remote session".
  # SessionInput doesn't declare the field, but google-adk v2's model config
  # allows extras (checked above via the version-agnostic probe).
  eval_case.session_input = SessionInput.model_construct(
      app_name=_REMOTE_APP_NAME,
      user_id='carol',
      state={'locale': 'en-US'},
      session_id='pre-existing-session',
  )
  eval_set = EvalSet(eval_set_id='weather_set', eval_cases=[eval_case])

  try:
    inference_results, eval_case_results = await _run_full_eval(
        service, eval_set, _METRICS
    )
  finally:
    await client.aclose()

  assert inference_results[0].status == InferenceStatus.SUCCESS
  assert inference_results[0].session_id == 'pre-existing-session'
  assert eval_case_results[0].final_eval_status == EvalStatus.PASSED

  # Never created (no POST .../sessions call) and never deleted.
  assert server.create_session_requests == []
  assert server.deleted_session_ids == []
  assert all(
      req['sessionId'] == 'pre-existing-session' for req in server.run_requests
  )


async def test_conversation_scenario_only_case_fails_without_stopping_others() -> (
    None
):
  server = FakeApiServer()
  server.scripts['dave'] = _weather_script()
  client = _client_for(server)
  eval_sets_manager = InMemoryEvalSetsManager()
  service = RemoteEvalService(
      client,
      app_name=_REMOTE_APP_NAME,
      eval_sets_manager=eval_sets_manager,
  )
  scenario_case = EvalCase(
      eval_id='scenario_case',
      conversation_scenario=ConversationScenario(
          starting_prompt='I need help with the weather.',
          conversation_plan='Ask about the weather in Tokyo, then thank the'
          ' agent.',
      ),
  )
  eval_set = EvalSet(
      eval_set_id='mixed_set',
      eval_cases=[scenario_case, _weather_eval_case('weather_case', 'dave')],
  )

  try:
    inference_results = await _register_and_perform_inference(
        service, eval_set
    )

    # perform_inference() is entirely RemoteEvalService's own code: this is
    # the core "one case failing doesn't stop the others" proof, and holds
    # regardless of google-adk version.
    results_by_id = {r.eval_case_id: r for r in inference_results}
    assert results_by_id['scenario_case'].status == InferenceStatus.FAILURE
    assert (
        'conversation_scenario' in results_by_id['scenario_case'].error_message
    )
    assert results_by_id['weather_case'].status == InferenceStatus.SUCCESS

    # The conversation_scenario case never touched the remote server at all;
    # only dave's (weather_case's) session was created and run.
    assert len(server.create_session_requests) == 1
    assert server.create_session_requests[0]['user_id'] == 'dave'
    assert len(server.run_requests) == 2  # dave's two turns only
    assert all(req['userId'] == 'dave' for req in server.run_requests)

    eval_case_results = (
        await _evaluate_tolerating_known_adk_v1_none_inferences_bug(
            service, inference_results, _METRICS
        )
    )
  finally:
    await client.aclose()

  eval_results_by_id = {r.eval_id: r for r in eval_case_results}
  assert eval_results_by_id['scenario_case'].final_eval_status == EvalStatus.FAILED
  assert eval_results_by_id['weather_case'].final_eval_status == EvalStatus.PASSED


async def test_http_error_for_one_case_does_not_stop_others() -> None:
  server = FakeApiServer()
  server.scripts['erin'] = _weather_script()
  server.scripts['frank'] = _weather_script()
  server.fail_user_ids.add('erin')
  client = _client_for(server)
  eval_sets_manager = InMemoryEvalSetsManager()
  service = RemoteEvalService(
      client,
      app_name=_REMOTE_APP_NAME,
      eval_sets_manager=eval_sets_manager,
  )
  eval_set = EvalSet(
      eval_set_id='mixed_set',
      eval_cases=[
          _weather_eval_case('failing_case', 'erin'),
          _weather_eval_case('ok_case', 'frank'),
      ],
  )

  try:
    inference_results = await _register_and_perform_inference(
        service, eval_set
    )

    # Again, entirely RemoteEvalService's own code and version-independent.
    results_by_id = {r.eval_case_id: r for r in inference_results}
    assert results_by_id['failing_case'].status == InferenceStatus.FAILURE
    assert results_by_id['failing_case'].error_message
    assert results_by_id['ok_case'].status == InferenceStatus.SUCCESS

    # The session created for the failing case was still cleaned up (created,
    # then deleted, even though the run itself failed).
    assert server.deleted_session_ids  # at least the failing case's session

    eval_case_results = (
        await _evaluate_tolerating_known_adk_v1_none_inferences_bug(
            service, inference_results, _METRICS
        )
    )
  finally:
    await client.aclose()

  eval_results_by_id = {r.eval_id: r for r in eval_case_results}
  assert eval_results_by_id['failing_case'].final_eval_status == EvalStatus.FAILED
  assert eval_results_by_id['ok_case'].final_eval_status == EvalStatus.PASSED
