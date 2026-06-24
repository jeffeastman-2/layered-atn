#!/usr/bin/env python3
"""
Unit tests for spatial_location validation in SpatialValidator.

Tests all spatial_location prepositions from the vocabulary to validate
the dot product logic in validate_spatial_relationship.
"""

import pytest
from latn.utils.spatial_validation import SpatialValidator
from latn.lexer.vector_space import VectorSpace
from latn.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE


class MockSceneObject:
    """Mock SceneObject for testing spatial validation."""
    
    def __init__(self, name: str, x: float, y: float, z: float):
        self.name = name
        self.position = {'x': x, 'y': y, 'z': z}
        self.vector = {
            'locX': x, 'locY': y, 'locZ': z,
            'scaleX': 1.0, 'scaleY': 1.0, 'scaleZ': 1.0
        }


@pytest.fixture
def test_objects():
    """Fixture providing test objects in a grid pattern."""
    return {
        # Reference object at origin
        'center': MockSceneObject("center", 0.0, 0.0, 0.0),
        
        # Objects positioned relative to center
        'above_obj': MockSceneObject("above", 0.0, 2.0, 0.0),     # +Y
        'below_obj': MockSceneObject("below", 0.0, -2.0, 0.0),    # -Y
        'right_obj': MockSceneObject("right", 2.0, 0.0, 0.0),     # +X
        'left_obj': MockSceneObject("left", -2.0, 0.0, 0.0),      # -X
        'front_obj': MockSceneObject("front", 0.0, 0.0, 2.0),     # +Z
        'behind_obj': MockSceneObject("behind", 0.0, 0.0, -2.0),  # -Z
        
        # Objects in wrong positions (should fail validation)
        'wrong_above': MockSceneObject("wrong_above", 0.0, -1.0, 0.0),  # Below when should be above
        'wrong_right': MockSceneObject("wrong_right", -1.0, 0.0, 0.0),   # Left when should be right
    }


def create_pp_token(prep_word: str) -> VectorSpace:
    """Get the preposition's vector directly from vocabulary."""
    if prep_word not in SEMANTIC_VECTOR_SPACE:
        raise ValueError(f"Preposition '{prep_word}' not found in vocabulary")
        
    # Just return the preposition's vector from vocabulary
    return SEMANTIC_VECTOR_SPACE[prep_word]


@pytest.mark.parametrize("prep", ['above', 'over'])
def test_above_relationships(test_objects, prep):
    """Test 'above' and 'over' prepositions (locY=1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: object is above reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj']], [test_objects['center']])
    assert score , f"'{prep}' should validate when object is above reference"
    
    # Incorrect: object is below reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['wrong_above']], [test_objects['center']])
    assert score == [False], f"'{prep}' should fail when object is below reference"


@pytest.mark.parametrize("prep", ['below', 'under'])
def test_below_relationships(test_objects, prep):
    """Test 'below' and 'under' prepositions (locY=-1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: object is below reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['below_obj']], [test_objects['center']])
    assert score, f"'{prep}' should validate when object is below reference"
    
    # Incorrect: object is above reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj']], [test_objects['center']])
    assert score == [False], f"'{prep}' should fail when object is above reference"


@pytest.mark.parametrize("prep", ['right of', 'right'])
def test_right_relationships(test_objects, prep):
    """Test 'right of' and 'right' prepositions (locX=1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: object is to the right of reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['right_obj']], [test_objects['center']])
    assert score , f"'{prep}' should validate when object is right of reference"
    
    # Incorrect: object is to the left of reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['wrong_right']], [test_objects['center']])
    assert score == [False], f"'{prep}' should fail when object is left of reference"

@pytest.mark.parametrize("prep", ['left of', 'left'])
def test_left_relationships(test_objects, prep):
    """Test 'left of' and 'left' prepositions (locX=-1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: object is to the left of reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['left_obj']], [test_objects['center']])
    assert score , f"'{prep}' should validate when object is left of reference"
    
    # Incorrect: object is to the right of reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['right_obj']], [test_objects['center']])
    assert score == [False], f"'{prep}' should fail when object is right of reference"

@pytest.mark.parametrize("prep", ['in front of', 'forward'])
def test_front_relationships(test_objects, prep):
    """Test 'in front of' and 'forward' prepositions (locZ=1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: object is in front of reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['front_obj']], [test_objects['center']])
    assert score , f"'{prep}' should validate when object is in front of reference"
    
    # Incorrect: object is behind reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['behind_obj']], [test_objects['center']])
    assert score == [False], f"'{prep}' should fail when object is behind reference"

@pytest.mark.parametrize("prep", ['behind', 'backward', 'back'])
def test_behind_relationships(test_objects, prep):
    """Test 'behind', 'backward', and 'back' prepositions (locZ=-1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: object is behind reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['behind_obj']], [test_objects['center']])
    assert score , f"'{prep}' should validate when object is behind reference"
    
    # Incorrect: object is in front of reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['front_obj']], [test_objects['center']])
    assert score == [False], f"'{prep}' should fail when object is in front of reference"

def test_up_down_relationships(test_objects):
    """Test 'up' and 'down' adverb/prepositions."""
    # Up (locY=1.0)
    pp_token = create_pp_token('up')
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj']], [test_objects['center']])
    assert score, "'up' should validate when object is above reference"
    
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['below_obj']], [test_objects['center']])
    assert score == [False], "'up' should fail when object is below reference"

    # Down (locY=-1.0)
    pp_token = create_pp_token('down')
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['below_obj']], [test_objects['center']])
    assert score, "'down' should validate when object is below reference"
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj']], [test_objects['center']])
    assert score == [False], "'down' should fail when object is above reference"


def test_dot_product_edge_cases(test_objects):
    """Test edge cases for dot product validation."""
    pp_token = create_pp_token('above')
    
    # Same position (dot product = 0) should fail
    same_pos = MockSceneObject("same", 0.0, 0.0, 0.0)
    score = SpatialValidator.validate_spatial_relationship(pp_token, [same_pos], [test_objects['center']])
    assert score == [False], "Same position should fail spatial validation"
    
    # Diagonal position (positive Y component) should pass
    diagonal_up = MockSceneObject("diagonal", 1.0, 1.0, 1.0)
    score = SpatialValidator.validate_spatial_relationship(pp_token, [diagonal_up], [test_objects['center']])
    assert score, "Diagonal up position should pass 'above' validation"
    
    # Diagonal position (negative Y component) should fail
    diagonal_down = MockSceneObject("diagonal", 1.0, -1.0, 1.0)
    score = SpatialValidator.validate_spatial_relationship(pp_token, [diagonal_down], [test_objects['center']])
    assert score == [False], "Diagonal down position should fail 'above' validation"


@pytest.mark.parametrize("prep", [
    'above', 'over', 'below', 'under', 'behind', 'in front of',
    'right of', 'left of', 'up', 'down', 'left', 'right', 
    'forward', 'backward', 'back'
])
def test_vocabulary_consistency(prep):
    """Test that all spatial_location prepositions can be processed."""
    # Should be able to create PP token without error
    pp_token = create_pp_token(prep)
    
    # Should have spatial_location feature
    assert pp_token.isa("spatial_location"), f"'{prep}' should have spatial_location feature"
    
    # Should have non-zero spatial vector component
    spatial_magnitude = abs(pp_token['locX']) + abs(pp_token['locY']) + abs(pp_token['locZ'])
    assert spatial_magnitude > 0.0, f"'{prep}' should have non-zero spatial vector"

@pytest.mark.parametrize("prep", ['above', 'over'])
def test_multiple_above_relationships(test_objects, prep):
    """Test 'above' and 'over' prepositions (locY=1.0)."""
    pp_token = create_pp_token(prep)
    
    # Correct: objects are above reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj'], test_objects['above_obj']], [test_objects['center']])
    assert score, f"'{prep}' should validate when objects are above reference"
    
    # Incorrect: objects are below reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['wrong_above'], test_objects['above_obj']], [test_objects['center']])
    assert score == [False, True], f"'{prep}' should fail when objects are below reference"

    # Correct: object is above references
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj']], [test_objects['center'], test_objects['center']])
    assert score, f"'{prep}' should validate when object is above all references"
    # Incorrect: object is below references
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj']], [test_objects['center'], test_objects['above_obj']])
    assert score == [False], f"'{prep}' should fail when object is below all references"

        # Correct: objects are above all reference
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['above_obj'], test_objects['above_obj']], [test_objects['center'], test_objects['center']])
    assert score, f"'{prep}' should validate when objects are above all references"
    # Incorrect: objects are below all references
    score = SpatialValidator.validate_spatial_relationship(pp_token, [test_objects['wrong_above'], test_objects['above_obj']], [test_objects['center'], test_objects['center']])
    assert score == [False, True], f"'{prep}' should fail when objects are below all references"

