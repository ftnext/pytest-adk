# Copyright 2026 pytest-adk contributors
"""RemoteEvalService: LocalEvalService with inference delegated to an api_server.

See ``REMOTE_EVAL_PLAN.md`` (sections 5, 6 task R-2, 7, 9) for the design
behind this module.

Import constraint: unlike ``client.py``, this module *does* need
``google.adk.evaluation.local_eval_service`` -- that is the whole point of
reusing ``LocalEvalService.evaluate()``. On google-adk v2, importing that
module transitively imports ``metric_evaluator_registry``, whose default
registry construction pulls in ``vertexai`` (google-cloud-aiplatform) even
though this module never talks to Vertex AI. The import is therefore guarded
below with a clear error message instead of a bare ``ModuleNotFoundError``,
mirroring ``google.adk.evaluation.agent_evaluator``'s own
``MISSING_EVAL_DEPENDENCIES_MESSAGE`` pattern.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.errors.not_found_error import NotFoundError
from google.adk.events import Event
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.evaluation.evaluation_generator import EvaluationGenerator

from .client import AdkApiClient

_MISSING_EVAL_DEPENDENCIES_MESSAGE = (
    "Remote evaluation (pytest_adk.remote.eval_service) reuses"
    " google.adk.evaluation.local_eval_service.LocalEvalService for scoring,"
    ' which requires google-adk\'s `eval` extra: `pip install "google-adk'
    '[eval]"`. On google-adk v2, `LocalEvalService`\'s import chain'
    " additionally requires `vertexai` (google-cloud-aiplatform) purely as an"
    " import-time side effect of its default metric registry -- this module"
    " never calls Vertex AI. If pytest-adk's own dependencies are already"
    " installed, adding the base `google-cloud-aiplatform` package on top is"
    " therefore sufficient without pulling in the rest of the `eval` extra."
)

try:
  from google.adk.evaluation.local_eval_service import LocalEvalService
except ModuleNotFoundError as e:
  raise ModuleNotFoundError(_MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

logger = logging.getLogger(__name__)

# Mirrors the ``EVAL_SESSION_ID_PREFIX`` convention ADK uses for locally
# generated eval sessions (historically ``cli/cli_eval.py``; as of google-adk
# 2.5.0 also defined in ``evaluation/local_eval_service.py``). Defined here
# independently -- rather than imported -- so this module never has to import
# ``google.adk.cli`` (which pulls in the whole FastAPI-based CLI stack) just
# for a string constant.
_EVAL_SESSION_ID_PREFIX = '___eval___session___'

_DUMMY_ROOT_AGENT_NAME = 'remote_eval_dummy_agent'


class RemoteEvalService(LocalEvalService):
  """LocalEvalService with inference delegated to a remote ``api_server``.

  Reuses ``LocalEvalService.evaluate()`` verbatim (metric scoring, parallel
  evaluation, rubric copying, result assembly) and overrides
  ``perform_inference()`` to run each eval case's conversation against a
  remote ADK ``api_server`` via :class:`~pytest_adk.remote.client.AdkApiClient`
  instead of an in-process ``Runner``.

  Why a dummy ``root_agent`` is safe: ``LocalEvalService`` stores whatever
  agent it is constructed with as ``self._root_agent``, and (verified against
  google-adk 2.5.0, ``google/adk/evaluation/local_eval_service.py``) that
  attribute is read in exactly one place in the whole class:
  ``perform_inference()``, where it is passed down to
  ``_perform_inference_single_eval_item()`` as the ``root_agent`` argument.
  ``evaluate()`` and everything it calls (``_evaluate_single_inference_result``,
  ``_evaluate_metric_for_eval_case``, ``_evaluate_metric``,
  ``_generate_final_eval_status``) never touch ``self._root_agent``. Because
  this class fully overrides ``perform_inference()`` and never calls
  ``super().perform_inference()``, the dummy agent passed to
  ``LocalEvalService.__init__`` is never actually run -- ``evaluate()`` scores
  the ``Invocation``s this class builds from remote HTTP responses, which is
  all it needs. This assumption is enforced by a guard test (see
  ``tests/remote/test_eval_service.py``, following the pattern of ROADMAP
  task 0-2) that scans ``LocalEvalService``'s other methods for
  ``self._root_agent`` and fails if a future google-adk release starts
  referencing it outside ``perform_inference()``.
  """

  def __init__(
      self,
      client: AdkApiClient,
      *,
      app_name: str,
      eval_sets_manager: EvalSetsManager,
      default_user_id: str = 'eval_user',
      eval_set_results_manager: EvalSetResultsManager | None = None,
      keep_sessions: bool = False,
  ) -> None:
    """Create a RemoteEvalService bound to a running ``api_server``.

    Args:
        client: Client used to talk to the remote ``api_server``.
        app_name: Name of the app on the *remote* server (passed to every
            ``AdkApiClient`` call). This is independent of the ``app_name``
            eval cases are namespaced under in ``eval_sets_manager`` --
            :class:`~google.adk.evaluation.base_eval_service.InferenceRequest`
            carries that separately, and it is typically an arbitrary local
            label (e.g. ``"test_app"``) rather than the remote app's name.
        eval_sets_manager: Source of ``EvalSet``/``EvalCase`` data, e.g.
            :class:`~google.adk.evaluation.in_memory_eval_sets_manager.InMemoryEvalSetsManager`.
        default_user_id: ``user_id`` used for a session when an eval case's
            ``session_input`` is absent or does not specify one.
        eval_set_results_manager: Optional persistence for ``EvalCaseResult``s,
            forwarded to ``LocalEvalService`` unchanged.
        keep_sessions: If ``True``, remote sessions created by this service
            are not deleted after use (useful for debugging). Sessions that
            already existed (an eval case's ``session_input`` specifies an
            explicit ``session_id``) are never deleted, regardless of this
            flag.
    """
    super().__init__(
        root_agent=BaseAgent(name=_DUMMY_ROOT_AGENT_NAME),
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
    )
    self._client = client
    self._app_name = app_name
    self._default_user_id = default_user_id
    self._keep_sessions = keep_sessions

  async def perform_inference(
      self,
      inference_request: InferenceRequest,
  ) -> AsyncGenerator[InferenceResult, None]:
    """Runs each selected eval case's conversation against the remote agent.

    Mirrors the structure of ``LocalEvalService.perform_inference()``
    (fetch the eval set, honor ``eval_case_ids`` filtering, bound concurrency
    with a semaphore sized by ``inference_config.parallelism``, yield results
    as they complete via ``asyncio.as_completed``) but drives each eval case
    through :meth:`_perform_remote_inference_single_eval_item` instead of an
    in-process ``Runner``.

    Args:
        inference_request: The request for generating inferences.

    Yields:
        One ``InferenceResult`` per selected eval case, as it completes.

    Raises:
        NotFoundError: If ``inference_request.eval_set_id`` is not found for
            ``inference_request.app_name`` in ``eval_sets_manager``.
    """
    eval_set = self._eval_sets_manager.get_eval_set(
        app_name=inference_request.app_name,
        eval_set_id=inference_request.eval_set_id,
    )
    if not eval_set:
      raise NotFoundError(
          f'Eval set with id {inference_request.eval_set_id} not found for'
          f' app {inference_request.app_name}'
      )

    eval_cases = eval_set.eval_cases
    if inference_request.eval_case_ids:
      eval_cases = [
          eval_case
          for eval_case in eval_cases
          if eval_case.eval_id in inference_request.eval_case_ids
      ]

    semaphore = asyncio.Semaphore(
        value=inference_request.inference_config.parallelism
    )

    async def run_inference(eval_case: EvalCase) -> InferenceResult:
      async with semaphore:
        return await self._perform_remote_inference_single_eval_item(
            app_name=inference_request.app_name,
            eval_set_id=inference_request.eval_set_id,
            eval_case=eval_case,
        )

    inference_results = [run_inference(eval_case) for eval_case in eval_cases]
    for inference_result in asyncio.as_completed(inference_results):
      yield await inference_result

  async def _perform_remote_inference_single_eval_item(
      self,
      *,
      app_name: str,
      eval_set_id: str,
      eval_case: EvalCase,
  ) -> InferenceResult:
    """Runs one eval case's conversation remotely and returns its result.

    Session handling: if ``eval_case.session_input`` carries an explicit
    ``session_id`` (an extra field -- ``SessionInput`` does not declare one.
    On google-adk v2 its model config allows extras, so a TOML/JSON evalset
    author can set one to point at a pre-provisioned remote session; on v1
    ``SessionInput`` forbids extras, so this only works when the eval case
    is constructed programmatically, e.g. via ``model_construct()`` -- see
    REMOTE_EVAL_PLAN.md sec 9 on version skew), that session is treated as
    already existing on the remote server: it is used as-is and never
    created or deleted here. Otherwise a fresh session is created with
    a generated id (``_EVAL_SESSION_ID_PREFIX`` + uuid4); the *server's*
    returned ``Session.id`` is then used for subsequent calls rather than the
    requested one, since older api_server versions may ignore the requested
    id. Unless ``keep_sessions`` is set, sessions created here are deleted in
    a ``finally`` block; a failed delete is logged, not raised, since it
    should not affect this eval case's already-computed result.

    Any error while resolving/creating the session or running a conversation
    turn (an HTTP error from ``AdkApiClient``, or an ``Event``/``Invocation``
    validation error) is caught so it only fails this eval case;
    ``perform_inference`` continues with the others.

    Args:
        app_name: The ``InferenceRequest.app_name`` this result belongs to
            (the local ``eval_sets_manager`` namespace, not the remote app).
        eval_set_id: ID of the eval set the eval case belongs to.
        eval_case: The eval case to run.

    Returns:
        An ``InferenceResult`` with ``status=SUCCESS`` and ``inferences`` set
        on success, or ``status=FAILURE`` and ``error_message`` set otherwise.
    """
    inference_result = InferenceResult(
        app_name=app_name,
        eval_set_id=eval_set_id,
        eval_case_id=eval_case.eval_id,
        session_id=None,
    )

    if eval_case.conversation is None:
      # Exactly one of `conversation` / `conversation_scenario` is set (see
      # EvalCase's model validator); `conversation is None` therefore means
      # this eval case only has a `conversation_scenario`, i.e. it expects a
      # UserSimulator to drive a dynamic multi-turn conversation. That is out
      # of scope for v1 remote evaluation (REMOTE_EVAL_PLAN.md sec 3) -- there
      # is no local agent loop to plug a simulator into against a bare HTTP
      # api_server.
      inference_result.status = InferenceStatus.FAILURE
      inference_result.error_message = (
          "RemoteEvalService does not support conversation_scenario-driven"
          " eval cases (user-simulator multi-turn) in v1; eval case"
          f" '{eval_case.eval_id}' has no static conversation."
      )
      return inference_result

    session_input = eval_case.session_input
    user_id = (
        session_input.user_id
        if session_input and session_input.user_id
        else self._default_user_id
    )
    initial_state = session_input.state if session_input else None
    requested_session_id = (
        getattr(session_input, 'session_id', None) if session_input else None
    )

    session_id = requested_session_id
    created_session = False
    try:
      if requested_session_id is None:
        session = await self._client.create_session(
            app_name=self._app_name,
            user_id=user_id,
            session_id=_EVAL_SESSION_ID_PREFIX + str(uuid.uuid4()),
            state=initial_state,
        )
        # Use the server's own id: older api_server versions may ignore the
        # requested session_id and assign their own.
        session_id = session.id
        created_session = True

      inference_result.session_id = session_id

      all_events: list[Event] = []
      for invocation in eval_case.conversation:
        turn_events = await self._client.run(
            app_name=self._app_name,
            user_id=user_id,
            session_id=session_id,
            new_message=invocation.user_content,
        )
        if turn_events:
          # The api_server's /run only returns agent-produced events, not a
          # "user" event for the message that was just sent -- but
          # EvaluationGenerator.convert_events_to_eval_invocations() derives
          # each Invocation's user_content from an author="user" event. This
          # synthesizes that event (reusing the turn's invocation_id, which
          # every response event for one /run call shares) so the resulting
          # Invocation carries the user_content we actually sent.
          all_events.append(
              Event(
                  invocation_id=turn_events[0].invocation_id,
                  author='user',
                  content=invocation.user_content,
              )
          )
        all_events.extend(turn_events)

      inference_result.inferences = (
          EvaluationGenerator.convert_events_to_eval_invocations(all_events)
      )
      inference_result.status = InferenceStatus.SUCCESS
    except Exception as e:  # noqa: BLE001 - isolate this eval case's failure
      # Any failure here (httpx.HTTPStatusError / ConnectError /
      # TimeoutException from AdkApiClient, or a pydantic ValidationError
      # while parsing an Event/Invocation) should only fail this eval case,
      # not the whole run -- mirrors LocalEvalService's own
      # `_perform_inference_single_eval_item`, which catches broadly for the
      # same reason.
      logger.warning(
          "Remote inference failed for eval case '%s': %s",
          eval_case.eval_id,
          e,
          exc_info=True,
      )
      inference_result.status = InferenceStatus.FAILURE
      inference_result.error_message = (
          f"Remote inference failed for eval case '{eval_case.eval_id}': {e}"
      )
    finally:
      if created_session and not self._keep_sessions:
        try:
          await self._client.delete_session(
              app_name=self._app_name,
              user_id=user_id,
              session_id=session_id,
          )
        except Exception as e:  # noqa: BLE001 - cleanup must not fail the case
          logger.warning(
              "Failed to delete eval session '%s' for eval case '%s': %s",
              session_id,
              eval_case.eval_id,
              e,
          )

    return inference_result
