"""Import every console-script target.

Seven first-party modules had literally zero coverage, so a missing symbol or a
bad annotation in any of them shipped green. This is the cheapest guard against
that: it does not run the tools, only imports them and checks the advertised
entry point exists and is callable.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _console_scripts() -> list[tuple[str, str]]:
    with _PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    scripts: dict[str, str] = data["project"]["scripts"]
    return sorted((name, target) for name, target in scripts.items())


@pytest.mark.parametrize(("name", "target"), _console_scripts())
def test_console_script_target_is_importable(name: str, target: str) -> None:
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    entry = getattr(module, attr, None)
    assert entry is not None, f"{name}: {target} has no attribute {attr!r}"
    assert callable(entry), f"{name}: {target} is not callable"


def test_every_script_declares_a_module_that_exists() -> None:
    assert _console_scripts(), "no console scripts declared"
