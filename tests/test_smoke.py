"""Smoke tests for the protea-method bootstrap."""

from __future__ import annotations

import protea_method


def test_version_is_string() -> None:
    assert isinstance(protea_method.__version__, str)


def test_no_platform_imports_leak() -> None:
    """Hard rule: importing protea_method must not pull in
    sqlalchemy / fastapi / protea-core."""
    import sys

    forbidden = {"sqlalchemy", "fastapi", "protea_core"}
    leaked = forbidden & set(sys.modules)
    assert not leaked, f"Forbidden modules leaked into sys.modules: {leaked}"
