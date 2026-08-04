"""Deferred, revision-aware noun-phrase grounding.

The parser may inspect grounding while constructing a sentence, but an
interpreter may execute several clauses against a changing world.  This
mapping-compatible promise lets both users share the same NP: parser code can
continue to call ``grounding.get(...)`` and an interpreter can call ``force``
again immediately before using the referent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional


def grounding_revision(adapter: object) -> Any:
    """Return an adapter's optional world/discourse revision token.

    Adapters are not required to implement revisioning.  An interpreter can
    still request ``force(refresh=True)`` at an execution boundary.
    """
    value = getattr(adapter, "grounding_revision", None)
    return value() if callable(value) else value


class GroundingPromise(dict):
    """A lazy grounding result with the old grounding-dict interface."""

    def __init__(self, adapter: object, resolver: Callable[[], dict]):
        super().__init__()
        self.adapter = adapter
        self._resolver = resolver
        self._forced = False
        self._revision: Optional[Any] = None

    @property
    def resolved(self) -> bool:
        self.force()
        return bool(dict.get(self, "scene_objects"))

    def force(self, *, refresh: bool = False) -> "GroundingPromise":
        revision = grounding_revision(self.adapter)
        stale = (
            not self._forced
            or refresh
            or (revision is not None and revision != self._revision)
        )
        if stale:
            result = self._resolver() or {}
            dict.clear(self)
            dict.update(self, result)
            self._forced = True
            self._revision = revision
        return self

    def invalidate(self) -> None:
        self._forced = False

    def get(self, key, default=None):
        self.force()
        return dict.get(self, key, default)

    def __getitem__(self, key):
        self.force()
        return dict.__getitem__(self, key)

    def __bool__(self):
        self.force()
        return dict.__len__(self) > 0

    def __deepcopy__(self, memo):
        # The adapter is an external live world and must not be snapshotted.
        clone = type(self)(self.adapter, self._resolver)
        if self._forced:
            dict.update(clone, self)
            clone._forced = True
            clone._revision = self._revision
        memo[id(self)] = clone
        return clone


def force_grounding(np, *, refresh: bool = False):
    """Force an NP grounding while remaining compatible with legacy dicts."""
    grounding = getattr(np, "grounding", None)
    if isinstance(grounding, GroundingPromise):
        return grounding.force(refresh=refresh)
    return grounding


__all__ = ["GroundingPromise", "force_grounding", "grounding_revision"]
