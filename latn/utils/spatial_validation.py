"""Host-policy delegation for prepositional-phrase relationships."""

from latn.lexer.spatial_policy import get_active_spatial_policy
from latn.lexer.vector_space import VectorSpace


class SpatialValidator:
    """Apply the active host policy across candidate entity pairs.

    The historical name is retained for API compatibility. LATN contains no
    coordinate system, geometry, distance threshold, or relation ontology.
    """

    @staticmethod
    def validate_spatial_relationship(pp_token, obj1s, obj2s) -> list[bool]:
        if not isinstance(pp_token, VectorSpace):
            return [False]
        policy = get_active_spatial_policy()
        results = []
        for obj1 in obj1s:
            results.append(all(policy.validate(pp_token, obj1, obj2) for obj2 in obj2s))
        return results


__all__ = ["SpatialValidator"]
