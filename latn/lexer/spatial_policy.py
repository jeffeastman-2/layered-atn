"""Host policy seam for validating parsed PP relationships."""

from typing import Optional, Protocol, runtime_checkable

from latn.lexer.scene_adapter import GroundedEntity
from latn.lexer.vector_space import VectorSpace


@runtime_checkable
class SpatialPolicy(Protocol):
    """Per-pair spatial-relationship verdict. The iteration over candidate
    object pairs stays in SpatialValidator.validate_spatial_relationship;
    this owns only the per-pair predicate."""

    def applies_to(self, pp_token: VectorSpace) -> bool:
        """Whether this host policy validates this parsed relationship."""
        ...

    def validate(
        self, pp_token: VectorSpace, obj1: GroundedEntity, obj2: GroundedEntity
    ) -> bool:
        ...


class PermissiveSpatialPolicy:
    """Accept every spatial relationship. For domains whose spatial
    semantics live entirely outside the LATN (e.g. resolved downstream);
    also the seam's test fixture."""

    def applies_to(self, pp_token: VectorSpace) -> bool:
        return False

    def validate(
        self, pp_token: VectorSpace, obj1: GroundedEntity, obj2: GroundedEntity
    ) -> bool:
        return True


_active: Optional[SpatialPolicy] = None


def get_active_spatial_policy() -> SpatialPolicy:
    """The policy SpatialValidator.validate_spatial_relationship consults.
    Call at use-time (not import-bound) so set/use_spatial_policy take
    effect."""
    global _active
    if _active is None:
        _active = PermissiveSpatialPolicy()
    return _active


def set_active_spatial_policy(policy: Optional[SpatialPolicy]) -> None:
    """Swap the active policy. Pass None to reset to the neutral default."""
    global _active
    _active = policy


class use_spatial_policy:
    """Context manager: activate `policy` for a scope, restore the prior
    one. For per-parse activation (Driftmoor) and hermetic tests."""

    def __init__(self, policy: SpatialPolicy):
        self._new = policy
        self._prev: Optional[SpatialPolicy] = None

    def __enter__(self) -> SpatialPolicy:
        global _active
        self._prev = _active
        _active = self._new
        return self._new

    def __exit__(self, *exc) -> None:
        global _active
        _active = self._prev


__all__ = [
    "SpatialPolicy",
    "PermissiveSpatialPolicy",
    "get_active_spatial_policy",
    "set_active_spatial_policy",
    "use_spatial_policy",
]
