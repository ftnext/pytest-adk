# Copyright 2026 pytest-adk contributors
"""Client for evaluating ADK agents exposed as ``adk api_server`` REST APIs.

See ``REMOTE_EVAL_PLAN.md`` for the design behind this package. ``client.py``
intentionally avoids importing ``google.adk.evaluation`` (and therefore
google-adk's ``eval`` extra) so it stays usable without pulling in pandas.

``eval_service.py`` (``RemoteEvalService``), by contrast, needs
``google.adk.evaluation.local_eval_service`` -- on google-adk v2 that
transitively needs ``vertexai`` (see ``eval_service.py``'s module docstring).
``RemoteEvalService`` is therefore exported lazily via ``__getattr__`` below
rather than imported at module load time, so that ``import pytest_adk.remote``
(and thus ``AdkApiClient``) keeps working even where that dependency chain is
unavailable; the requirement only bites once ``RemoteEvalService`` is actually
accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .client import AdkApiClient

if TYPE_CHECKING:
  from .eval_service import RemoteEvalService as RemoteEvalService

__all__ = [
    'AdkApiClient',
    'RemoteEvalService',
]


def __getattr__(name: str):
  """Lazily import ``RemoteEvalService`` (see module docstring for why)."""
  if name == 'RemoteEvalService':
    from .eval_service import RemoteEvalService

    return RemoteEvalService
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
