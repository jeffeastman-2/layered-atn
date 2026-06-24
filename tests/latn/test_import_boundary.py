"""Enforced import boundary for the standalone LATN package.

LATN was extracted from Engraf; nothing in the ``latn`` package may import its
former host (or any ``engraf`` module). This keeps the parser host-agnostic:
grounding reaches the world only through the injected ``SceneAdapter``, never by
importing a specific host. (Docstrings/comments that mention engraf in prose are
fine — this checks actual ``import``/``from`` statements.)
"""

import re
from pathlib import Path

import latn

FORBIDDEN = re.compile(r"^\s*(?:from|import)\s+engraf(?:\.|\s|$)", re.MULTILINE)
LATN_ROOT = Path(latn.__file__).parent


def test_latn_package_does_not_import_engraf():
    offenders = []
    for path in LATN_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        m = FORBIDDEN.search(path.read_text())
        if m:
            rel = path.relative_to(LATN_ROOT).as_posix()
            offenders.append(f"{rel}: {m.group(0).strip()}")
    assert not offenders, "latn must not import engraf:\n" + "\n".join(offenders)
