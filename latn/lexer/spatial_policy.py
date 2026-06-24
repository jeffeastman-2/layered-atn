"""Injectable Layer-3 spatial-relationship policy.

Phase-1 seam 4 of factoring the LATN core out of Engraf. The per-pair
"is obj1 in the spatial relationship `pp` to obj2?" verdict was baked
into SpatialValidator.validate_single_relationship, with a hardcoded
proximity threshold (`distance < 1.0`) in arbitrary scene units. That
"near = 1 unit" rule is a first-pass placeholder that is meaningless
for any domain not using Engraf's coordinate scale (e.g. Driftmoor,
whose world is in feet).

Both spatial predicates -- the dot-product sign test for spatial_location
(above/below/behind/...) and the distance-threshold for spatial_proximity
(near/at/in/on) -- now live behind a SpatialPolicy. Driftmoor can swap
in its own (zone-aware, LOS-aware, etc.) without touching the LATN.

EngrafSpatialPolicy is the prior validate_single_relationship body moved
verbatim -> behavior-preserving by default (threshold=1.0). The proximity
threshold is a constructor parameter for trivial overrides; for non-
distance proximity semantics, write a different SpatialPolicy.
"""

from typing import Optional, Protocol, runtime_checkable

from latn.lexer.scene_adapter import GroundedEntity
from latn.lexer.vector_space import VectorSpace


@runtime_checkable
class SpatialPolicy(Protocol):
    """Per-pair spatial-relationship verdict. The iteration over candidate
    object pairs stays in SpatialValidator.validate_spatial_relationship;
    this owns only the per-pair predicate."""

    def validate(
        self, pp_token: VectorSpace, obj1: GroundedEntity, obj2: GroundedEntity
    ) -> bool:
        ...


class EngrafSpatialPolicy:
    """Engraf's default spatial predicates. Verbatim move of the former
    SpatialValidator.validate_single_relationship, including its defensive
    broad-except (preserves behavior on malformed inputs).

    `proximity_threshold` is in the world's coordinate units; the default
    1.0 matches Engraf's prior baked-in literal. Driftmoor in feet might
    use `EngrafSpatialPolicy(proximity_threshold=30.0)`.
    """

    def __init__(self, proximity_threshold: float = 1.0):
        self.proximity_threshold = proximity_threshold

    def validate(
        self, pp_token: VectorSpace, obj1: GroundedEntity, obj2: GroundedEntity
    ) -> bool:
        try:
            pos1 = obj1.position
            pos2 = obj2.position

            dx, dy, dz = pos1['x'] - pos2['x'], pos1['y'] - pos2['y'], pos1['z'] - pos2['z']
            px = pp_token['locX']
            py = pp_token['locY']
            pz = pp_token['locZ']

            if pp_token.isa("spatial_location"):
                # Sign test: is obj1 on the side of obj2 indicated by the
                # preposition's direction vector? Domain-neutral.
                dot = (dx * px + dy * py + dz * pz)
                return True if dot > 0 else False
            elif pp_token.isa("spatial_proximity"):
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                return True if distance < self.proximity_threshold else False
            else:
                return False
        except Exception:
            return False


class PermissiveSpatialPolicy:
    """Accept every spatial relationship. For domains whose spatial
    semantics live entirely outside the LATN (e.g. resolved downstream);
    also the seam's test fixture."""

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
        _active = EngrafSpatialPolicy()
    return _active


def set_active_spatial_policy(policy: Optional[SpatialPolicy]) -> None:
    """Swap the active policy. Pass None to reset to the Engraf default."""
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
    "EngrafSpatialPolicy",
    "PermissiveSpatialPolicy",
    "get_active_spatial_policy",
    "set_active_spatial_policy",
    "use_spatial_policy",
]
