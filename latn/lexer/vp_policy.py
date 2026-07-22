"""Host policy seam for Layer-4 verb-phrase validation."""

from typing import Optional, Protocol, runtime_checkable

from latn.pos.verb_phrase import VerbPhrase


@runtime_checkable
class VPGroundingPolicy(Protocol):
    """Decides whether a verb phrase is semantically well-formed for the
    active domain. The grounder owns hypothesis iteration; this owns the
    per-VP verdict."""

    def validate_vp(self, vp: VerbPhrase) -> bool:
        ...


class PermissiveVPPolicy:
    """Accept every verb phrase. For domains (e.g. Driftmoor) whose verb
    semantics are adjudicated downstream rather than by L4's CAD rules;
    also the seam's test fixture."""

    def validate_vp(self, vp: VerbPhrase) -> bool:
        return True


_active: Optional[VPGroundingPolicy] = None


def get_active_vp_policy() -> VPGroundingPolicy:
    """The policy L4 consults. Call at use-time (not import-bound) so
    set_active_vp_policy / use_vp_policy take effect."""
    global _active
    if _active is None:
        _active = PermissiveVPPolicy()
    return _active


def set_active_vp_policy(policy: Optional[VPGroundingPolicy]) -> None:
    """Swap the active policy. Pass None to reset to the neutral default."""
    global _active
    _active = policy


class use_vp_policy:
    """Context manager: activate `policy` for a scope, restore the prior
    one. For per-parse activation (Driftmoor) and hermetic tests."""

    def __init__(self, policy: VPGroundingPolicy):
        self._new = policy
        self._prev: Optional[VPGroundingPolicy] = None

    def __enter__(self) -> VPGroundingPolicy:
        global _active
        self._prev = _active
        _active = self._new
        return self._new

    def __exit__(self, *exc) -> None:
        global _active
        _active = self._prev


__all__ = [
    "VPGroundingPolicy",
    "PermissiveVPPolicy",
    "get_active_vp_policy",
    "set_active_vp_policy",
    "use_vp_policy",
]
