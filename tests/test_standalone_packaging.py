"""E3/T-110 regression guard: standalone installable dependency declarations.

scholar-rag-kit used to declare its sibling Nexus Scholar kits as BARE names
(``"scholar-graph-kit"``) that were only resolvable through the relative
editable ``[tool.uv.sources]`` entries pointing at ``../scholar-graph-kit`` and
friends. Those entries only resolved inside the harness monorepo or beside a
sibling checkout on disk, so the published wheel could not be installed
standalone anywhere else.

This test locks in the fix: every sibling must be a PEP 508 direct git
reference pinned to a full 40-hex canonical SHA, and no relative path source may
come back. It is hermetic -- it reads the checked-in ``pyproject.toml`` only and
never touches the network. When a wheel has already been built into ``dist/``,
the built METADATA is additionally checked.
"""

from __future__ import annotations

import re
import tomllib
import zipfile
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# scholar-<name>[@ git+https://github.com/nexus-scholar-org/scholar-<name>@<40-hex>]
# with an optional PEP 508 extras suffix, e.g. scholar-pdf-kit[extract].
DIRECT_REF = re.compile(
    r"^scholar-[a-z0-9-]+(\[[a-z0-9,.-]+\])? @ git\+https://github\.com/nexus-scholar-org/scholar-[a-z0-9-]+@[0-9a-f]{40}$"
)

# The canonical main SHAs these siblings are pinned to. Tracked to the merged
# canonical mains after bib#1/graph#1 merged at 21:50Z (T-110-REPIN); the
# structural DIRECT_REF regex and the full-40-hex requirement are unchanged, so
# this mirror tracks the pins rather than relaxing what the guard enforces.
EXPECTED_SHA = {
    "scholar-protocol-kit": "4e10f25c25a1b150ce518348d211c7771683a9b7",
    "scholar-graph-kit": "646b84cec215ebd9b7b449448bca879492e78492",
    "scholar-bib-kit": "fbdd38ba25301e613621f18119623f04d2673dca",
    "scholar-search-kit": "911d864fcb6a706d4c0339f80524a46f591e2cad",
}


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_every_sibling_is_a_sha_pinned_direct_git_reference() -> None:
    deps = _load()["project"]["dependencies"]
    siblings = [d for d in deps if d.startswith("scholar-")]
    assert siblings, "expected declared scholar-* sibling requirements"

    for dep in siblings:
        assert DIRECT_REF.match(dep), f"not a SHA-pinned direct git reference: {dep!r}"

    declared = {re.split(r"[ @\[]", d, maxsplit=1)[0]: d for d in siblings}
    assert set(declared) == set(EXPECTED_SHA), f"unexpected sibling set: {sorted(declared)}"
    for name, sha in EXPECTED_SHA.items():
        assert declared[name].endswith(sha), f"{name} is not pinned to canonical main {sha}: {declared[name]!r}"


def test_no_relative_editable_sibling_source_can_come_back() -> None:
    sources = _load().get("tool", {}).get("uv", {}).get("sources", {})
    relative = {name: src for name, src in sources.items() if isinstance(src, dict) and "path" in src}
    assert not relative, f"relative sibling sources break standalone installs: {relative}"


def test_built_wheel_metadata_carries_the_direct_refs() -> None:
    wheels = sorted((PYPROJECT.parent / "dist").glob("*.whl"))
    if not wheels:
        import pytest

        pytest.skip("no built wheel in dist/; run `uv build --wheel .` first")

    metadata_name = next(n for n in zipfile.ZipFile(wheels[-1]).namelist() if n.endswith(".dist-info/METADATA"))
    requires = [
        line.removeprefix("Requires-Dist: ")
        for line in zipfile.ZipFile(wheels[-1]).read(metadata_name).decode().splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    git_requires = [r for r in requires if "git+" in r]
    assert len(git_requires) == len(EXPECTED_SHA), f"expected {len(EXPECTED_SHA)} git direct refs, got {git_requires}"
    for name, sha in EXPECTED_SHA.items():
        assert any(name in r and r.endswith(sha) for r in git_requires), f"missing {name}@{sha} in METADATA"
