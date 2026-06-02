# Copyright 2026 pytest-adk contributors
"""Helpers to resume an exported ADK session JSON with an in-memory Runner."""

from __future__ import annotations

from pathlib import Path

from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session


def load_session_from_json(path_or_str: str | Path) -> Session:
  """Load an exported session from a file path or raw JSON string.

  Args:
      path_or_str: Path to a ``.session.json``-style file, or the JSON text.
          A ``str`` whose (stripped) first character is ``{`` or ``[`` is
          treated as JSON content; otherwise it is treated as a file path and
          must exist.

  Returns:
      Parsed :class:`google.adk.sessions.session.Session`.

  Raises:
      FileNotFoundError: If ``path_or_str`` refers to a path that does not
          exist on disk.
  """
  if isinstance(path_or_str, Path):
    if not path_or_str.is_file():
      raise FileNotFoundError(path_or_str)
    data = path_or_str.read_text(encoding='utf-8')
  else:
    stripped = path_or_str.lstrip()
    # Avoid Path(...).is_file() on large JSON strings (long path errors).
    if stripped.startswith('{') or stripped.startswith('['):
      data = path_or_str
    else:
      candidate = Path(path_or_str)
      try:
        is_file = candidate.is_file()
      except OSError as exc:
        raise FileNotFoundError(path_or_str) from exc
      if not is_file:
        raise FileNotFoundError(path_or_str)
      data = candidate.read_text(encoding='utf-8')

  return Session.model_validate_json(data)


async def runner_from_exported_session(
    agent: BaseAgent,
    exported: Session | str | Path,
    *,
    app_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    artifact_service: BaseArtifactService | None = None,
    memory_service: BaseMemoryService | None = None,
    credential_service: BaseCredentialService | None = None,
) -> tuple[Runner, Session]:
  """Build an in-memory :class:`Runner` whose session service holds the export.

  This mirrors the ``saved_session_file`` flow in ``google.adk.cli``: create a
  fresh session with copied ``state``, then :meth:`append_event` for each export
  event. Call :meth:`Runner.run_async` with the same ``user_id`` and
  ``session.id`` to continue the conversation.

  Args:
      agent: Root agent for the runner (must match the exported app logically).
      exported: Parsed session, or a path / JSON string for
          :func:`load_session_from_json`.
      app_name: Override runner app name (default: from export).
      user_id: Override user id (default: from export).
      session_id: Override session id (default: from export; keeps ids aligned).
      artifact_service: Optional; defaults to :class:`InMemoryArtifactService`.
      memory_service: Optional; defaults to :class:`InMemoryMemoryService`.
      credential_service: Optional; defaults to
          :class:`InMemoryCredentialService`.

  Returns:
      ``(runner, session)`` where ``session`` is the restored in-memory session
      (also reachable via ``runner.session_service.get_session``).

  Raises:
      FileNotFoundError: If ``exported`` is a path-like value that cannot be
          loaded by :func:`load_session_from_json`.
  """
  if not isinstance(exported, Session):
    exported = load_session_from_json(exported)

  resolved_app_name = app_name or exported.app_name
  resolved_user_id = user_id or exported.user_id
  resolved_session_id = session_id or exported.id

  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name=resolved_app_name,
      user_id=resolved_user_id,
      state=exported.state,
      session_id=resolved_session_id,
  )

  for event in exported.events:
    await session_service.append_event(session, event)

  runner = Runner(
      app_name=resolved_app_name,
      agent=agent,
      session_service=session_service,
      artifact_service=artifact_service
      if artifact_service is not None
      else InMemoryArtifactService(),
      memory_service=memory_service
      if memory_service is not None
      else InMemoryMemoryService(),
      credential_service=credential_service
      if credential_service is not None
      else InMemoryCredentialService(),
  )

  return runner, session
