"""Contract test: the L3 spatial verdict comes from the *active*
SpatialPolicy, not a hardcoded threshold/predicate.

The existing tests/latn/layer3/test_spatial_location_validation.py covers
behavior preservation of the default Engraf rules end-to-end through
SpatialValidator.validate_spatial_relationship; this file proves the
seam itself -- injection, scoping, parameterized threshold, and the
fail-closed default vs the permissive override.
"""

from latn.lexer.spatial_policy import (
    StrictSpatialPolicy,
    PermissiveSpatialPolicy,
    SpatialPolicy,
    set_active_spatial_policy,
    use_spatial_policy,
)
from latn.lexer.vector_space import VectorSpace, vector_from_features
from latn.utils.spatial_validation import SpatialValidator


class _Pt:
    """Minimal GroundedEntity-shaped stub. SpatialPolicy only reads .position."""

    def __init__(self, x, y, z, name="pt", oid="pt"):
        self.position = {"x": x, "y": y, "z": z}
        self.name = name
        self.vector = VectorSpace()
        self.object_id = oid
        self.entity_id = oid


def _proximity_pp():
    # Distance uses object positions, not pp loc values; isa("spatial_proximity")
    # is what selects the proximity branch.
    return vector_from_features("prep spatial_proximity")


def _above_pp():
    # Direction +Y selects the spatial_location dot-test.
    return vector_from_features("prep spatial_location", loc=[0.0, 1.0, 0.0])


def test_policies_satisfy_protocol():
    assert isinstance(StrictSpatialPolicy(), SpatialPolicy)
    assert isinstance(PermissiveSpatialPolicy(), SpatialPolicy)


def test_engraf_default_proximity_threshold_is_one_unit():
    pol = StrictSpatialPolicy()
    near = _Pt(0.5, 0, 0)
    far = _Pt(2.0, 0, 0)
    center = _Pt(0, 0, 0)
    assert pol.validate(_proximity_pp(), near, center)       # 0.5 < 1.0
    assert not pol.validate(_proximity_pp(), far, center)    # 2.0 >= 1.0


def test_proximity_threshold_is_parameterizable_for_other_unit_systems():
    # Driftmoor in feet would do StrictSpatialPolicy(proximity_threshold=30.0).
    pol = StrictSpatialPolicy(proximity_threshold=30.0)
    obj_25ft = _Pt(25.0, 0, 0)
    center = _Pt(0, 0, 0)
    assert pol.validate(_proximity_pp(), obj_25ft, center)


def test_spatial_location_dot_test_is_unchanged_by_threshold():
    # The dot-test is domain-neutral and ignores proximity_threshold.
    pol = StrictSpatialPolicy(proximity_threshold=0.001)
    above = _Pt(0, 5.0, 0)
    below = _Pt(0, -5.0, 0)
    center = _Pt(0, 0, 0)
    assert pol.validate(_above_pp(), above, center)
    assert not pol.validate(_above_pp(), below, center)


def test_injection_flows_through_validate_spatial_relationship_and_is_scoped():
    far = _Pt(50.0, 0, 0)
    center = _Pt(0, 0, 0)
    pp = _proximity_pp()

    # Default (Engraf, threshold 1.0): far object rejected.
    assert SpatialValidator.validate_spatial_relationship(pp, [far], [center]) == [False]

    # Injected permissive policy: same pair accepted.
    with use_spatial_policy(PermissiveSpatialPolicy()):
        assert SpatialValidator.validate_spatial_relationship(pp, [far], [center]) == [True]

    # Restored: default again rejects.
    assert SpatialValidator.validate_spatial_relationship(pp, [far], [center]) == [False]


def test_set_active_spatial_policy_reset():
    far = _Pt(50.0, 0, 0)
    center = _Pt(0, 0, 0)
    pp = _proximity_pp()
    try:
        set_active_spatial_policy(PermissiveSpatialPolicy())
        assert SpatialValidator.validate_spatial_relationship(pp, [far], [center]) == [True]
    finally:
        set_active_spatial_policy(None)  # reset to Engraf default
    assert SpatialValidator.validate_spatial_relationship(pp, [far], [center]) == [False]
