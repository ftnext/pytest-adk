# Copyright 2026 pytest-adk contributors
"""Shared pytest configuration for the test suite.

``pytester`` is enabled so plugin-level behavior (e.g. the
``pytest_adk_prompt_template_engine`` ini option) can be exercised end to end in
an isolated, temporary pytest project.
"""

from __future__ import annotations

pytest_plugins = ['pytester']
