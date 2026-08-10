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
from typing import Any
from typing import AsyncGenerator
from typing import Sequence

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
from google.adk.sessions import Session

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
  # Same guard, same reason: on google-adk v2 this is the very module whose
  # default-registry construction needs `vertexai`, so importing it outside
  # the guard would turn the friendly message above back into a bare
  # ModuleNotFoundError.
  from google.adk.evaluation.metric_evaluator_registry import (
      MetricEvaluatorRegistry,
  )
except ModuleNotFoundError as e:
  raise ModuleNotFoundError(_MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

logger = logging.getLogger(__name__)

_DUMMY_ROOT_AGENT_NAME = 'remote_eval_dummy_agent'


def _pinned_session_id(eval_case: EvalCase) -> str | None:
  """Returns the existing remote session an eval case pins, if any.

  ``session_id`` is not a declared ``SessionInput`` field. It survives only on
  google-adk v2, whose model config allows extra fields (v1 forbids them and
  drops it), hence ``getattr`` rather than attribute access.

  An empty string counts as *not* pinned. TOML has no null literal, so
  ``session_id = ""`` is the only way an evalset author can spell "no pinned
  session" for a field they would otherwise have to omit entirely; treating it
  as a pin would mean running the conversation against a session literally
  named ``''`` and never cleaning it up.

  This is the single source of truth for "does this eval case reuse a
  session?", shared by the runtime path here and by ``pytest-adk eval``'s
  ``--num-runs`` guard, so the two cannot drift apart.

  A value that is present but neither a string nor ``None`` (google-adk v2
  keeps extra fields without validating their types, so ``session_id = 123``
  in a TOML evalset arrives here as an ``int``) is rejected rather than
  quietly treated as unpinned: silently creating a fresh session would run the
  eval case against different state and history than the evalset asked for.

  Returns:
      The pinned session ID, or ``None`` when the eval case should get a
      freshly created session.

  Raises:
      ValueError: If ``session_id`` is set to a non-string value.
  """
  session_input = eval_case.session_input
  if session_input is None:
    return None
  session_id = getattr(session_input, 'session_id', None)
  if session_id is None:
    return None
  if not isinstance(session_id, str):
    raise ValueError(
        f"Eval case '{eval_case.eval_id}' has a non-string"
        f' session_input.session_id ({session_id!r}, type'
        f' {type(session_id).__name__}). Quote it to pin an existing remote'
        ' session, or remove it to get a fresh session per run.'
    )
  # An empty string is the documented "not pinned" spelling (see above), not
  # an error: unlike a wrong type, it is the only way TOML can express it.
  return session_id or None


def _group_eval_cases_by_pinned_session(
    eval_cases: Sequence[EvalCase], default_user_id: str
) -> dict[tuple[str, str], list[str]]:
  """Groups eval case ids by the remote session they pin.

  Two eval cases address the same remote session only when *both* the
  resolved user and the session id match, so the key is that pair. Cases that
  pin nothing are not grouped.

  A case whose ``session_id`` has a bad type is skipped rather than raised on:
  it already fails on its own inside the per-case boundary, and raising here
  would abort the whole grouping (and, from ``perform_inference``, every other
  case's result).

  This is the single source of truth for "which eval cases share a session",
  used by the service to fail the conflicting cases and by ``pytest-adk
  eval``'s preflight to reject the run outright, so the two cannot disagree
  about what counts as sharing.

  Returns:
      ``{(user_id, session_id): [eval_case_id, ...]}``; a list longer than one
      entry is a conflict.
  """
  by_session: dict[tuple[str, str], list[str]] = {}
  for eval_case in eval_cases:
    try:
      session_id = _pinned_session_id(eval_case)
    except ValueError:
      continue
    if session_id is None:
      continue
    key = (_resolved_user_id(eval_case, default_user_id), session_id)
    by_session.setdefault(key, []).append(eval_case.eval_id)
  return by_session


def _shared_pinned_session_errors(
    eval_cases: Sequence[EvalCase], default_user_id: str
) -> dict[str, str]:
  """Maps each eval case that shares a pinned session to its error message.

  Returns:
      ``{eval_case_id: message}``, empty when every pin is unique.
  """
  errors: dict[str, str] = {}
  groups = _group_eval_cases_by_pinned_session(eval_cases, default_user_id)
  for (user_id, session_id), case_ids in groups.items():
    if len(case_ids) < 2:
      continue
    for eval_case_id in case_ids:
      others = [i for i in case_ids if i != eval_case_id]
      errors[eval_case_id] = (
          f"Eval case '{eval_case_id}' pins the same remote session (user_id"
          f' {user_id!r}, session_input.session_id {session_id!r}) as'
          f' {", ".join(repr(i) for i in others)}. They would share one'
          " mutable session and contaminate each other's turns, state changes"
          ' and tool side effects, so none of them is run. Give each eval case'
          ' its own session_input.session_id, or drop it so each gets a fresh'
          ' session.'
      )
  return errors


def _pinned_session_state_error(
    eval_case: EvalCase, pinned_session_id: str | None
) -> str | None:
  """Reports an eval case that pins a session *and* declares initial state.

  The two are mutually exclusive in practice: a pinned session is used as-is
  and never created, so ``session_input.state`` is never sent anywhere and the
  conversation runs against whatever state the pre-existing session already
  holds. Silently ignoring a declared starting condition can only produce
  scores for something other than what the evalset describes, so it is
  rejected.

  ``state`` defaults to ``{}`` (a pydantic ``default_factory``), not ``None``,
  so only a *non-empty* mapping counts as declared -- an author who simply
  omitted the field must not trip this.

  Shared by the runtime and by ``pytest-adk eval``'s preflight so the two
  cannot drift apart, in the manner of :func:`_pinned_session_id`.

  Args:
      eval_case: The eval case to check.
      pinned_session_id: Its already-resolved pinned session id, or ``None``.

  Returns:
      ``None`` when there is no conflict, otherwise a ready-to-print message.
  """
  if pinned_session_id is None:
    return None
  session_input = eval_case.session_input
  state = getattr(session_input, 'state', None) if session_input else None
  if not state:
    return None
  return (
      f"Eval case '{eval_case.eval_id}' both pins an existing remote session"
      f' (session_input.session_id {pinned_session_id!r}) and declares'
      ' session_input.state. A pinned session is used as-is and never'
      ' created, so that state would never be applied and the conversation'
      ' would run against whatever the session already holds. Drop the'
      ' session_id to start from the declared state, or drop the state to'
      " accept the pinned session's own."
  )


def _resolved_user_id(eval_case: EvalCase, default_user_id: str) -> str:
  """Returns the user an eval case's remote session belongs to.

  An eval case's own ``session_input.user_id`` wins; otherwise the service's
  default (``pytest-adk eval``'s ``--user-id``) applies.

  Like :func:`_pinned_session_id`, this is shared with the CLI rather than
  duplicated there: a pinned session is only *the same* session when both the
  user and the session id match, so the guard has to resolve the user exactly
  as the runtime does.
  """
  session_input = eval_case.session_input
  if session_input is not None and session_input.user_id:
    return session_input.user_id
  return default_user_id


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
      metric_evaluator_registry: MetricEvaluatorRegistry | None = None,
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
        metric_evaluator_registry: Registry ``evaluate()`` resolves metric
            names through, forwarded to ``LocalEvalService`` unchanged.
            ``None`` leaves ADK's process-wide default in place, which knows
            only the built-in metrics; pass
            :func:`pytest_adk.metrics.build_metric_evaluator_registry`'s
            result to also score an ``EvalConfig``'s ``custom_metrics``.
    """
    super().__init__(
        root_agent=BaseAgent(name=_DUMMY_ROOT_AGENT_NAME),
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
        metric_evaluator_registry=metric_evaluator_registry,
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

    Note on repeated calls: an eval case that pins an existing remote session
    (``session_input.session_id``, google-adk v2 only) is *not* isolated
    across calls. Every call sends the conversation to that same mutable
    server-side session, so a second call sees the first call's turns, state
    changes and tool side effects rather than repeating it independently.
    Callers that repeat inference to average over runs should either use one
    call per pinned-session eval case or let each run create a fresh session
    by leaving ``session_id`` unset. ``pytest-adk eval`` drives that repeat
    loop itself and rejects ``--num-runs > 1`` for such eval cases; this is a
    documented property here rather than a runtime check because this method
    has no visibility into how often the caller intends to invoke it.

    Args:
        inference_request: The request for generating inferences.

    Yields:
        One ``InferenceResult`` per selected eval case, as it completes.

    Raises:
        NotFoundError: If ``inference_request.eval_set_id`` is not found for
            ``inference_request.app_name`` in ``eval_sets_manager``.
        ValueError: If ``inference_request.inference_config.parallelism`` is
            not >= 1 -- ``asyncio.Semaphore`` accepts 0 or negative values
            without error, but then never lets any task acquire it, which
            would otherwise deadlock this method forever instead of failing
            fast.
    """
    if inference_request.inference_config.parallelism < 1:
      raise ValueError(
          'inference_request.inference_config.parallelism must be >= 1,'
          f' got {inference_request.inference_config.parallelism!r}.'
      )

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

    # Cross-case, so it cannot be decided inside a single eval case's run:
    # two cases pinning one session contaminate each other whether they run
    # concurrently or in sequence. Computed before scheduling, and reported as
    # a FAILURE for the conflicting cases *only* -- raising here would stop
    # the generator and deny results to every unrelated case, which is the
    # opposite of this method's per-case isolation contract.
    shared_session_errors = _shared_pinned_session_errors(
        eval_cases, self._default_user_id
    )

    async def run_inference(eval_case: EvalCase) -> InferenceResult:
      error_message = shared_session_errors.get(eval_case.eval_id)
      if error_message is not None:
        return InferenceResult(
            app_name=inference_request.app_name,
            eval_set_id=inference_request.eval_set_id,
            eval_case_id=eval_case.eval_id,
            session_id=None,
            status=InferenceStatus.FAILURE,
            error_message=error_message,
        )
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
    created or deleted here. Otherwise a fresh session is created by
    :meth:`_create_eval_session`, and the *server's* returned ``Session.id``
    -- no ``sessionId`` is ever requested -- is what subsequent calls use.
    Unless ``keep_sessions`` is set, sessions created here are deleted in
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

    if not eval_case.conversation:
      # A schema-valid `conversation = []`. Left alone, the turn loop below
      # would make no /run calls at all, convert an empty event list into
      # empty inferences, and still report SUCCESS -- handing scoring a
      # result produced without ever contacting the deployed agent, whose
      # empty-aggregate metric behavior can then look like a verdict. There
      # is nothing to evaluate, so say so instead.
      inference_result.status = InferenceStatus.FAILURE
      inference_result.error_message = (
          f"Eval case '{eval_case.eval_id}' has an empty conversation: there"
          ' are no turns to send to the remote agent, so nothing can be'
          ' inferred or scored.'
      )
      return inference_result

    session_input = eval_case.session_input
    user_id = _resolved_user_id(eval_case, self._default_user_id)
    initial_state = session_input.state if session_input else None

    session_id: str | None = None
    created_session = False
    try:
      # Resolved *inside* the failure boundary: _pinned_session_id() validates
      # this eval case's own session_id and raises on a malformed one. Above
      # the try, that ValueError would escape perform_inference() entirely and
      # stop the async generator, so one bad eval case would deny results to
      # every other (valid) case running alongside it -- per-case isolation is
      # this method's contract. ``pytest-adk eval`` also rejects the same
      # input up front, but direct RemoteEvalService callers bypass that.
      requested_session_id = _pinned_session_id(eval_case)
      session_id = requested_session_id

      state_error = _pinned_session_state_error(eval_case, requested_session_id)
      if state_error is not None:
        raise ValueError(state_error)

      if requested_session_id is None:
        session = await self._create_eval_session(
            user_id=user_id, initial_state=initial_state
        )
        # The server assigns the id; no sessionId was sent in the request.
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
        # The api_server's /run only returns agent-produced events, not a
        # "user" event for the message that was just sent -- but
        # EvaluationGenerator.convert_events_to_eval_invocations() derives
        # each Invocation's user_content from an author="user" event. This
        # synthesizes that event (reusing the turn's invocation_id, which
        # every response event for one /run call shares) so the resulting
        # Invocation carries the user_content we actually sent.
        #
        # This happens even when the server returned no events at all: that
        # turn still has to become an Invocation, otherwise it silently
        # vanishes and every later turn shifts up by one, so scoring would
        # compare each expectation against the wrong turn's response. With the
        # synthesized user event alone, the turn converts to an Invocation
        # carrying user_content and no final_response -- an unanswered prompt,
        # which the metrics can score (and fail) on its own terms. A fresh
        # invocation_id is used because there is no event to take one from.
        all_events.append(
            Event(
                invocation_id=(
                    turn_events[0].invocation_id
                    if turn_events
                    else str(uuid.uuid4())
                ),
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

  async def _create_eval_session(
      self, *, user_id: str, initial_state: dict[str, Any] | None
  ) -> Session:
    """Creates a temporary session for one eval case.

    Never sends a client-supplied ``sessionId``: some session services reject
    one outright, e.g. a deployment that assigns ids in its own format
    (issue #10), or a Vertex AI-backed ``api_server``, which answers an
    ADK-style ``___eval___session___...`` id with HTTP 500 rather than a 4xx
    (issue #15) -- so there is no status code that reliably distinguishes "id
    rejected" from "server broken" to retry on. The very same request without
    a ``sessionId`` succeeds on every server observed so far, and the
    server's returned ``Session.id`` is authoritative regardless, so nothing
    is lost by never asking.

    Args:
        user_id: User to create the session for.
        initial_state: Initial session state, or ``None`` to send none.

    Returns:
        The created :class:`~google.adk.sessions.Session`.

    Raises:
        httpx.HTTPStatusError: If the server rejects the create request.
    """
    return await self._client.create_session(
        app_name=self._app_name,
        user_id=user_id,
        state=initial_state,
    )
