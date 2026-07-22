"""Spatial meaning is supplied by a host policy, not LATN dimensions."""

from latn.lexer.spatial_policy import (
    PermissiveSpatialPolicy,
    SpatialPolicy,
    set_active_spatial_policy,
    use_spatial_policy,
)
from latn.lexer.vector_space import vector_from_features
from latn.utils.spatial_validation import SpatialValidator


class RejectSpatialPolicy:
    def applies_to(self, pp_token):
        return True

    def validate(self, pp_token, obj1, obj2):
        return False


class Entity:
    position = {"x": 0.0, "y": 0.0, "z": 0.0}


def _relation():
    return vector_from_features("prep")


def test_policies_satisfy_protocol():
    assert isinstance(PermissiveSpatialPolicy(), SpatialPolicy)
    assert isinstance(RejectSpatialPolicy(), SpatialPolicy)


def test_default_is_neutral_and_custom_policy_is_scoped():
    args = (_relation(), [Entity()], [Entity()])
    assert SpatialValidator.validate_spatial_relationship(*args) == [True]

    with use_spatial_policy(RejectSpatialPolicy()):
        assert SpatialValidator.validate_spatial_relationship(*args) == [False]

    assert SpatialValidator.validate_spatial_relationship(*args) == [True]


def test_set_active_policy_reset_restores_neutral_default():
    args = (_relation(), [Entity()], [Entity()])
    try:
        set_active_spatial_policy(RejectSpatialPolicy())
        assert SpatialValidator.validate_spatial_relationship(*args) == [False]
    finally:
        set_active_spatial_policy(None)
    assert SpatialValidator.validate_spatial_relationship(*args) == [True]
