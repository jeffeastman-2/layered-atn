"""
Tests for comparative and superlative adjective vector dimensions.

This module tests the "comp" and "super" vector dimensions that were added
to distinguish between comparative forms (bigger, redder) and superlative
forms (biggest, reddest) in the semantic vector space.
"""

import pytest
from latn.lexer.vocabulary_builder import vector_from_word
from latn.utils.adjective_inflector import base_adjective_from_comparative
from latn.lexer.vector_space import VectorSpace, VECTOR_DIMENSIONS


class TestComparativeSuperlativeDetection:
    """Test detection and classification of comparative/superlative forms."""
    
    def test_base_adjective_from_comparative_function(self):
        """Test the base_adjective_from_comparative function returns correct form types."""
        # Test comparative forms (-er endings)
        base, form_type = base_adjective_from_comparative("bigger")
        assert base == "big"
        assert form_type == "comparative"
        
        base, form_type = base_adjective_from_comparative("redder")
        assert base == "red"
        assert form_type == "comparative"
        
        base, form_type = base_adjective_from_comparative("taller")
        assert base == "tall"
        assert form_type == "comparative"
        
        base, form_type = base_adjective_from_comparative("rougher")
        assert base == "rough"
        assert form_type == "comparative"
        
        # Test superlative forms (-est endings)
        base, form_type = base_adjective_from_comparative("biggest")
        assert base == "big"
        assert form_type == "superlative"
        
        base, form_type = base_adjective_from_comparative("reddest")
        assert base == "red"
        assert form_type == "superlative"
        
        base, form_type = base_adjective_from_comparative("tallest")
        assert base == "tall"
        assert form_type == "superlative"
        
        base, form_type = base_adjective_from_comparative("roughest")
        assert base == "rough"
        assert form_type == "superlative"
        
        # Test base forms (not comparative/superlative)
        base, form_type = base_adjective_from_comparative("red")
        assert base == "red"
        assert form_type == "base"
        
        base, form_type = base_adjective_from_comparative("tall")
        assert base == "tall"
        assert form_type == "base"


class TestComparativeVectorDimensions:
    """Test that comparative adjectives set the 'comp' dimension correctly."""
    
    def test_comparative_adjectives_set_comp_dimension(self):
        """Test that comparative adjectives have comp=1.0 and super=0.0."""
        comparative_adjectives = ["bigger", "redder", "taller", "rougher", "smoother"]
        
        for adj in comparative_adjectives:
            vector = vector_from_word(adj)
            assert vector["comp"] == 1.0, f"{adj} should have comp=1.0, got {vector['comp']}"
            assert vector["super"] == 0.0, f"{adj} should have super=0.0, got {vector['super']}"
            assert vector["adj"] == 1.0, f"{adj} should be marked as adjective"
    
    def test_comparative_scaling_multiplier(self):
        """Test that comparative adjectives get 1.2x scaling boost."""
        # Test with a color adjective that has a base color value
        red_vector = vector_from_word("red")
        redder_vector = vector_from_word("redder")
        
        # Redder should have 1.2x the red value of red
        expected_red_value = red_vector["red"] * 1.2
        assert redder_vector["red"] == pytest.approx(expected_red_value, abs=1e-6), \
            f"redder should have red={expected_red_value}, got {redder_vector['red']}"
        
        # Test with a scale adjective
        big_vector = vector_from_word("big")
        bigger_vector = vector_from_word("bigger")
        
        # Bigger should have 1.2x the scale values of big
        for scale_dim in ["scaleX", "scaleY", "scaleZ"]:
            expected_scale = big_vector[scale_dim] * 1.2
            assert bigger_vector[scale_dim] == pytest.approx(expected_scale, abs=1e-6), \
                f"bigger should have {scale_dim}={expected_scale}, got {bigger_vector[scale_dim]}"


class TestSuperlativeVectorDimensions:
    """Test that superlative adjectives set the 'super' dimension correctly."""
    
    def test_superlative_adjectives_set_super_dimension(self):
        """Test that superlative adjectives have super=1.0 and comp=0.0."""
        superlative_adjectives = ["biggest", "reddest", "tallest", "roughest", "smoothest"]
        
        for adj in superlative_adjectives:
            vector = vector_from_word(adj)
            assert vector["super"] == 1.0, f"{adj} should have super=1.0, got {vector['super']}"
            assert vector["comp"] == 0.0, f"{adj} should have comp=0.0, got {vector['comp']}"
            assert vector["adj"] == 1.0, f"{adj} should be marked as adjective"
    
    def test_superlative_scaling_multiplier(self):
        """Test that superlative adjectives get 1.5x scaling boost."""
        # Test with a color adjective that has a base color value
        red_vector = vector_from_word("red")
        reddest_vector = vector_from_word("reddest")
        
        # Reddest should have 1.5x the red value of red
        expected_red_value = red_vector["red"] * 1.5
        assert reddest_vector["red"] == pytest.approx(expected_red_value, abs=1e-6), \
            f"reddest should have red={expected_red_value}, got {reddest_vector['red']}"
        
        # Test with a scale adjective
        big_vector = vector_from_word("big")
        biggest_vector = vector_from_word("biggest")
        
        # Biggest should have 1.5x the scale values of big
        for scale_dim in ["scaleX", "scaleY", "scaleZ"]:
            expected_scale = big_vector[scale_dim] * 1.5
            assert biggest_vector[scale_dim] == pytest.approx(expected_scale, abs=1e-6), \
                f"biggest should have {scale_dim}={expected_scale}, got {biggest_vector[scale_dim]}"


class TestBaseAdjectiveVectorDimensions:
    """Test that base adjectives don't set comp or super dimensions."""
    
    def test_base_adjectives_no_comp_super(self):
        """Test that base adjectives have comp=0.0 and super=0.0."""
        base_adjectives = ["red", "big", "tall", "rough", "smooth", "blue", "green"]
        
        for adj in base_adjectives:
            vector = vector_from_word(adj)
            assert vector["comp"] == 0.0, f"{adj} should have comp=0.0, got {vector['comp']}"
            assert vector["super"] == 0.0, f"{adj} should have super=0.0, got {vector['super']}"
            assert vector["adj"] == 1.0, f"{adj} should be marked as adjective"


class TestComparativeSuperlativeDifferentialScaling:
    """Test the differential scaling between comparative and superlative forms."""
    
    def test_differential_scaling_comparison(self):
        """Test that superlatives are scaled more than comparatives."""
        # Test with texture adjectives
        rough_vector = vector_from_word("rough")
        rougher_vector = vector_from_word("rougher")
        roughest_vector = vector_from_word("roughest")
        
        base_texture = rough_vector["texture"]
        comparative_texture = rougher_vector["texture"]
        superlative_texture = roughest_vector["texture"]
        
        # Verify the scaling relationships
        assert comparative_texture == pytest.approx(base_texture * 1.2, abs=1e-6), \
            "Comparative should be 1.2x base"
        assert superlative_texture == pytest.approx(base_texture * 1.5, abs=1e-6), \
            "Superlative should be 1.5x base"
        assert superlative_texture > comparative_texture, \
            "Superlative should be scaled more than comparative"
        
        # Test with transparency adjectives
        clear_vector = vector_from_word("clear")
        # Note: We don't have "clearer/clearest" in vocabulary, so test with transparent
        transparent_vector = vector_from_word("transparent")
        
        # Verify base transparency value exists
        assert transparent_vector["transparency"] > 0, \
            "Transparent should have positive transparency value"


class TestVectorSpaceDimensionsExist:
    """Test that comp and super dimensions exist in the vector space."""
    
    def test_comp_super_in_vector_dimensions(self):
        """Test that 'comp' and 'super' are in VECTOR_DIMENSIONS."""
        assert "comp" in VECTOR_DIMENSIONS, "'comp' dimension should exist in VECTOR_DIMENSIONS"
        assert "super" in VECTOR_DIMENSIONS, "'super' dimension should exist in VECTOR_DIMENSIONS"
    
    def test_comp_super_accessible_in_vector_space(self):
        """Test that comp and super dimensions are accessible in VectorSpace objects."""
        vs = VectorSpace()
        
        # Should be able to get and set comp/super values
        vs["comp"] = 1.0
        vs["super"] = 1.0
        
        assert vs["comp"] == 1.0, "Should be able to set and get comp dimension"
        assert vs["super"] == 1.0, "Should be able to set and get super dimension"
        
        # Test with fresh vector space
        vs2 = VectorSpace()
        assert vs2["comp"] == 0.0, "New VectorSpace should have comp=0.0 by default"
        assert vs2["super"] == 0.0, "New VectorSpace should have super=0.0 by default"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_irregular_comparatives_superlatives(self):
        """Test handling of irregular comparative/superlative forms."""
        # Test irregular comparative forms
        irregular_comparatives = {
            "better": "good",
            "worse": "bad", 
            "more": "much",
            "further": "far",
            "farther": "far"
        }
        
        for word, expected_base in irregular_comparatives.items():
            base, form_type = base_adjective_from_comparative(word)
            assert base == expected_base, f"{word} should map to {expected_base}"
            assert form_type == "comparative", f"{word} should be identified as comparative"
        
        # Test irregular superlative forms
        irregular_superlatives = {
            "best": "good",
            "worst": "bad",
            "most": "much", 
            "furthest": "far",
            "farthest": "far"
        }
        
        for word, expected_base in irregular_superlatives.items():
            base, form_type = base_adjective_from_comparative(word)
            assert base == expected_base, f"{word} should map to {expected_base}"
            assert form_type == "superlative", f"{word} should be identified as superlative"
    
    def test_unknown_comparative_adjectives(self):
        """Test that unknown comparative adjectives raise appropriate errors."""
        with pytest.raises(ValueError, match="Unknown token"):
            vector_from_word("nonexistenter")  # Not in vocabulary
        
        with pytest.raises(ValueError, match="Unknown token"):
            vector_from_word("nonexistentest")  # Not in vocabulary
    
    def test_case_insensitive_comparatives(self):
        """Test that comparative detection is case insensitive."""
        # Test uppercase
        base, form_type = base_adjective_from_comparative("BIGGER")
        assert base == "big"
        assert form_type == "comparative"
        
        # Test mixed case
        base, form_type = base_adjective_from_comparative("TaLlEsT")
        assert base == "tall"
        assert form_type == "superlative"


class TestIntegrationWithExistingSystem:
    """Test integration with existing adjective processing system."""
    
    def test_comparative_adjectives_still_adjectives(self):
        """Test that comparative adjectives are still recognized as adjectives."""
        comparative_adjectives = ["bigger", "redder", "taller"]
        
        for adj in comparative_adjectives:
            vector = vector_from_word(adj)
            # Should still be marked as adjective
            assert vector.isa("adj"), f"{adj} should be recognized as adjective"
            assert vector["adj"] == 1.0, f"{adj} should have adj=1.0"
    
    def test_superlative_adjectives_still_adjectives(self):
        """Test that superlative adjectives are still recognized as adjectives."""
        superlative_adjectives = ["biggest", "reddest", "tallest"]
        
        for adj in superlative_adjectives:
            vector = vector_from_word(adj)
            # Should still be marked as adjective
            assert vector.isa("adj"), f"{adj} should be recognized as adjective"
            assert vector["adj"] == 1.0, f"{adj} should have adj=1.0"
    
    def test_word_attribute_preserved(self):
        """Test that the word attribute is preserved in comparative/superlative vectors."""
        vector = vector_from_word("bigger")
        assert vector.word == "bigger", "Word attribute should be preserved"
        
        vector = vector_from_word("biggest")
        assert vector.word == "biggest", "Word attribute should be preserved"
