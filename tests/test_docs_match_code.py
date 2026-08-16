"""Catch documentation that describes code the repository does not have.

Every doc-vs-code divergence found so far was cheap to detect and expensive to
find by reading: ``docs/`` described a ``k(N, beta2)`` factor for months after it
was deleted, the SF derivations use ``sf_lr``/``sf_beta1`` identifiers that have
never existed in ``src/``, and ``README.md`` promised plots from an entry point
that no longer draws any.

These checks are deliberately narrow. They look only at things with one
unambiguous machine-checkable form -- config attribute paths, registry names,
console-script names, relative links -- and say nothing about prose. A broad
"every backticked token must exist" check would drown in false positives and get
turned off.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import OPTIMIZER_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = sorted((ROOT / "docs").glob("*.md")) + [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
]

#: `SPSAConfig.foo.bar` / `config.foo.bar` references inside backticks.
CONFIG_PATH = re.compile(r"`(?:SPSAConfig|config)\.([a-z_][a-z0-9_.]*)`")

#: Quoted optimizer names, the form docs/Simulator.md uses for registry entries.
REGISTRY_NAME = re.compile(r'`"([a-z0-9-]+)"`')

#: `uv run <script>` invocations.
UV_RUN = re.compile(r"uv run (?:--group dev )?([a-z0-9-]+)")

#: `config.py`, `config.toml` and friends match CONFIG_PATH but name files.
FILE_SUFFIXES = frozenset({"py", "toml", "md", "json", "yaml", "yml", "lock"})

#: Words that look like registry names but are not; keep this list short.
NOT_OPTIMIZERS = frozenset({"outcomes", "const-mean", "const_mean", "text", "python"})


def _console_scripts() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return set(tomllib.load(handle)["project"]["scripts"])


def _resolve_attr_path(root: object, dotted: str) -> bool:
    current = root
    for part in dotted.split("."):
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    return True


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_config_paths_exist(doc: Path) -> None:
    """Every `SPSAConfig.x.y` a doc names must resolve on a real config."""
    config = SPSAConfig()
    missing = sorted(
        {
            path
            for path in CONFIG_PATH.findall(doc.read_text())
            if path not in FILE_SUFFIXES
            and not _resolve_attr_path(config, path.rstrip("."))
        },
    )
    assert not missing, (
        f"{doc.name} names config attributes that do not exist: {missing}"
    )


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_quoted_optimizer_names_are_registered(doc: Path) -> None:
    """A doc that quotes an optimizer name must quote one that exists."""
    text = doc.read_text()
    candidates = {
        name
        for name in REGISTRY_NAME.findall(text)
        if name not in NOT_OPTIMIZERS
        and ("spsa" in name or "adam" in name or "sgd" in name)
    }
    missing = sorted(candidates - set(OPTIMIZER_REGISTRY))
    assert not missing, f"{doc.name} names unregistered optimizers: {missing}"


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_uv_run_targets_exist(doc: Path) -> None:
    """`uv run <name>` must name a console script or a dev tool."""
    dev_tools = {"pytest", "ruff", "ty", "pre-commit", "python", "sync"}
    known = _console_scripts() | dev_tools
    missing = sorted(set(UV_RUN.findall(doc.read_text())) - known)
    assert not missing, f"{doc.name} invokes unknown entry points: {missing}"


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_relative_links_resolve(doc: Path) -> None:
    """Eight links were broken at once by a missing `../`; keep them honest."""
    broken: list[str] = []
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
        link = target.strip()
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = link.split("#")[0]
        if path and not (doc.parent / path).exists():
            broken.append(link)
    assert not broken, f"{doc.name} has broken relative links: {sorted(set(broken))}"


def test_every_registry_entry_is_documented() -> None:
    """docs/Simulator.md must mention every optimizer the lab ships."""
    text = (ROOT / "docs" / "Simulator.md").read_text()
    undocumented = sorted(
        name for name in OPTIMIZER_REGISTRY if f'"{name}"' not in text
    )
    assert not undocumented, f"docs/Simulator.md omits: {undocumented}"


def test_every_console_script_is_in_the_readme() -> None:
    """README.md is the entry-point index; it must list all of them."""
    text = (ROOT / "README.md").read_text()
    missing = sorted(name for name in _console_scripts() if name not in text)
    assert not missing, f"README.md omits console scripts: {missing}"


def test_the_k_factor_is_gone_from_the_docs() -> None:
    """The damping factor is retired; only its obituary may remain.

    docs/SF_Adam_derivation.md and docs/Simulator.md each keep one paragraph
    explaining why there is no such factor. Anything beyond that is a page that
    has drifted back to describing code the lab does not run.
    """
    offenders = {}
    for doc in DOC_FILES:
        hits = doc.read_text().count("k(N, beta2)") + doc.read_text().count("k(N, β2)")
        if hits > 2:  # noqa: PLR2004
            offenders[doc.name] = hits
    assert not offenders, (
        f"k(N, beta2) is described as live in: {offenders}. "
        "SFAdamBlock applies no such factor."
    )
