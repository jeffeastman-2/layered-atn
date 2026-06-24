"""
Shared spatial validation utilities for PP attachment and object positioning.

This module factors out common spatial reasoning logic used by both:
1. Layer 3 semantic grounding for PP attachment validation
2. Object modifier for spatial positioning calculations

Key patterns factored out:
- Preposition vector interpretation (locX, locY, locZ)
- Object dimension calculations
- Spatial relationship validation logic
- Position calculation based on object bounds

The per-pair spatial verdict for L3 grounding is supplied by the active
SpatialPolicy (latn.lexer.spatial_policy); validate_spatial_relationship
below owns only the obj1s x obj2s iteration. The interpreter-side helpers
(get_object_half_scale, extract_direction_factors, calculate_spatial_position)
remain here and are out of scope for the LATN extraction.
"""

from typing import Tuple, Optional, Union
from latn.lexer.spatial_policy import get_active_spatial_policy
from latn.lexer.vector_space import VectorSpace
from latn.pos.prepositional_phrase import PrepositionalPhrase
from latn.pos.noun_phrase import NounPhrase


class SpatialValidator:
    """Shared spatial validation and calculation utilities."""

    @staticmethod
    def get_object_half_scale(obj) -> Tuple[float, float, float]:
        """Get the half-scale of an object based on its type and size.

        Args:
            obj: SceneObject with vector containing scale dimensions

        Returns:
            tuple: (half_height, half_breadth, half_depth) representing the object's half-dimensions
        """
        if hasattr(obj, 'name') and 'cube' in obj.name.lower():
            # For cubes, all scales represent edge lengths, so half-size is scale/2
            half_height = obj.vector['scaleY'] / 2.0
            half_breadth = obj.vector['scaleX'] / 2.0
            half_depth = obj.vector['scaleZ'] / 2.0
        elif hasattr(obj, 'name') and 'sphere' in obj.name.lower():
            # For spheres, all scales represent radius, so half-size equals the radius
            radius = max(obj.vector['scaleX'], obj.vector['scaleY'], obj.vector['scaleZ'])
            half_height = radius
            half_breadth = radius
            half_depth = radius
        else:
            # Default: assume scales represent full dimensions, so half-size is scale/2
            half_height = obj.vector['scaleY'] / 2.0
            half_breadth = obj.vector['scaleX'] / 2.0
            half_depth = obj.vector['scaleZ'] / 2.0

        return half_height, half_breadth, half_depth

    @staticmethod
    def extract_direction_factors(preposition_vector) -> Tuple[float, float, float]:
        """Extract directional factors from a preposition vector.

        Args:
            preposition_vector: VectorSpace object containing locX, locY, locZ values

        Returns:
            tuple: (x_factor, y_factor, z_factor) representing spatial direction
        """
        x_factor = preposition_vector['locX'] if 'locX' in preposition_vector and preposition_vector['locX'] != 0.0 else 0.0
        y_factor = preposition_vector['locY'] if 'locY' in preposition_vector and preposition_vector['locY'] != 0.0 else 0.0
        z_factor = preposition_vector['locZ'] if 'locZ' in preposition_vector and preposition_vector['locZ'] != 0.0 else 0.0

        return x_factor, y_factor, z_factor

    @staticmethod
    def calculate_spatial_position(moving_obj, ref_obj, preposition_vector) -> Tuple[float, float, float]:
        """Calculate the position for spatial relationships like 'above', 'below', 'beside', etc.

        Uses the preposition vector to determine spatial direction and object dimensions
        for proper spacing.

        Args:
            moving_obj: Object being positioned
            ref_obj: Reference object for spatial relationship
            preposition_vector: VectorSpace with locX, locY, locZ direction factors

        Returns:
            tuple: (new_x, new_y, new_z) position for the moving object
        """
        # Get direction factors from the preposition vector
        x_factor, y_factor, z_factor = SpatialValidator.extract_direction_factors(preposition_vector)

        # Get reference object's position and size
        ref_x = ref_obj.vector['locX']
        ref_y = ref_obj.vector['locY']
        ref_z = ref_obj.vector['locZ']

        # Calculate object dimensions for proper spacing
        refHeight, refBreadth, refDepth = SpatialValidator.get_object_half_scale(ref_obj)
        movingHeight, movingBreadth, movingDepth = SpatialValidator.get_object_half_scale(moving_obj)

        # Start with reference object's position as base
        new_x = ref_x
        new_y = ref_y
        new_z = ref_z

        # Calculate X position based on directional factor
        if x_factor > 0:
            # Place object to the positive X direction (right/beside)
            new_x = ref_x + refBreadth + movingBreadth + abs(x_factor)
        elif x_factor < 0:
            # Place object to the negative X direction (left)
            new_x = ref_x - refBreadth - movingBreadth - abs(x_factor)

        # Calculate Y position based on directional factor
        if y_factor > 0:
            # Place object in positive Y direction (above)
            new_y = ref_y + refHeight + movingHeight + abs(y_factor)
        elif y_factor < 0:
            # Place object in negative Y direction (below)
            new_y = ref_y - refHeight - movingHeight - abs(y_factor)

        # Calculate Z position based on directional factor
        if z_factor > 0:
            # Place object in positive Z direction (behind)
            new_z = ref_z + refDepth + movingDepth + abs(z_factor)
        elif z_factor < 0:
            # Place object in negative Z direction (in front)
            new_z = ref_z - refDepth - movingDepth - abs(z_factor)

        return new_x, new_y, new_z

    @staticmethod
    def validate_spatial_relationship(pp_token, obj1s, obj2s) -> list[bool]:
        """Validate a spatial relationship between two objects using the active SpatialPolicy.

        Args:
            pp_token: PP token containing spatial features (VectorSpace with locX, locY, locZ) or preposition string
            obj1s: Reference objects (obj2s are positioned relative to obj1s)
            obj2s: Objects being positioned relative to obj1s
        Returns:
            list[bool]: Validation for each obj1 against all obj2s
        """
        if not isinstance(pp_token, VectorSpace):
            return [False]
        policy = get_active_spatial_policy()
        results = []
        for o1 in obj1s:
            all_valid = True
            for o2 in obj2s:
                if not policy.validate(pp_token, o1, o2):
                    all_valid = False
                    break
            results.append(all_valid)
        return results
