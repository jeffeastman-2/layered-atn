#!/usr/bin/env python3
"""
Unit tests for enhanced NP grammar supporting LATN architecture.

Tests the enhanced noun phrase parser that can handle:
- Bare nouns: "sphere"
- Adjective-initial NPs: "red sphere", "big red sphere"  
- Adverb-initial NPs: "very big sphere", "extremely red sphere"
- Traditional determiner-initial NPs: "the red sphere"

This supports LATN's multi-hypothesis generation where NPs can be found
starting at any token position.
"""

import pytest
from latn.lexer.token_stream import TokenStream
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1  # Use full LATN tokenization
from latn.atn.subnet_np import run_np
from latn.lexer.vector_space import VectorSpace


def parse_np_with_latn(text):
    """Parse an NP using full LATN multi-hypothesis tokenization.
    
    This reflects real usage where LATN generates multiple tokenization hypotheses
    and tries to parse NPs from each one.
    """
    tokenization_hypotheses = latn_tokenize_layer1(text)
    
    for hypothesis in tokenization_hypotheses:
        if hypothesis.tokens:  # Valid token list
            token_stream = TokenStream(hypothesis.tokens)
            np_result = run_np(token_stream)
            if np_result:
                return np_result  # Return first successful parse
    
    return None  # No successful parse found


class TestEnhancedNPGrammar:
    """Test cases for the enhanced NP grammar supporting LATN architecture."""
    
    def test_bare_noun_np(self):
        """Test NPs that are just bare nouns."""
        test_cases = [
            "sphere",
            "box", 
            "cylinder"
        ]
        
        for case in test_cases:
            result = parse_np_with_latn(case)
            
            assert result is not None, f"Failed to parse bare noun: '{case}'"
            assert result.noun == case, f"Expected noun '{case}', got '{result.noun}'"
            assert result.determiner is None, f"Bare noun should have no determiner"
            assert case in result.get_consumed_words(), f"Should consume the noun token"
            
            # Check vector has noun component
            assert result.vector["noun"] > 0, f"Vector should have noun component"
    
    def test_adjective_initial_nps(self):
        """Test NPs that start with adjectives."""
        test_cases = [
            ("red sphere", ["red", "sphere"]),
            ("big box", ["big", "box"]),
            ("small cylinder", ["small", "cylinder"])
        ]

        for case, expected_tokens in test_cases:
            hypotheses = latn_tokenize_layer1(case)
            tokens = hypotheses[0].tokens  # Use best hypothesis
            result = run_np(TokenStream(tokens))
            
            assert result is not None, f"Failed to parse adjective-initial NP: '{case}'"
            assert result.get_consumed_words() == expected_tokens, f"Expected tokens {expected_tokens}, got {result.get_consumed_words()}"
            assert result.determiner is None, f"Adjective-initial NP should have no determiner"
            
            # Check vector has both adjective and noun components
            assert result.vector["adj"] > 0, f"Vector should have adjective component"
            assert result.vector["noun"] > 0, f"Vector should have noun component"
    
    def test_multiple_adjective_nps(self):
        """Test NPs with multiple adjectives."""
        test_cases = [
            ("big red sphere", ["big", "red", "sphere"]),
            ("small blue box", ["small", "blue", "box"]),
            ("tall green cylinder", ["tall", "green", "cylinder"])
        ]

        for case, expected_tokens in test_cases:
            hypotheses = latn_tokenize_layer1(case)
            tokens = hypotheses[0].tokens  # Use best hypothesis
            result = run_np(TokenStream(tokens))
            
            assert result is not None, f"Failed to parse multi-adjective NP: '{case}'"
            assert result.get_consumed_words() == expected_tokens, f"Expected tokens {expected_tokens}, got {result.get_consumed_words()}"
            
            # Check vector has adjective components (should be > 1 for multiple adjectives)
            assert result.vector["adj"] > 1, f"Vector should have multiple adjective components"
            assert result.vector["noun"] > 0, f"Vector should have noun component"
    
    def test_adverb_initial_nps(self):
        """Test NPs that start with adverbs modifying adjectives."""
        test_cases = [
            ("very big sphere", ["very", "big", "sphere"]),
            ("extremely red box", ["extremely", "red", "box"]),
            ("very small cylinder", ["very", "small", "cylinder"])  # Changed from "really" to "very"
        ]

        for case, expected_tokens in test_cases:
            hypotheses = latn_tokenize_layer1(case)
            tokens = hypotheses[0].tokens  # Use best hypothesis
            result = run_np(TokenStream(tokens))
            
            assert result is not None, f"Failed to parse adverb-initial NP: '{case}'"
            assert result.get_consumed_words() == expected_tokens, f"Expected tokens {expected_tokens}, got {result.get_consumed_words()}"
            assert result.determiner is None, f"Adverb-initial NP should have no determiner"
            
            # Check that adverb scaling was applied (adjective value should be scaled)
            assert result.vector["adj"] > 1, f"Adjective should be scaled by adverb"
            assert result.vector["noun"] > 0, f"Vector should have noun component"
    
    def test_adverb_scaling_effects(self):
        """Test that adverbs properly scale adjectives."""
        # Parse "big sphere" (baseline)
        hypotheses1 = latn_tokenize_layer1("big sphere")
        tokens1 = hypotheses1[0].tokens  # Use best hypothesis
        result1 = run_np(TokenStream(tokens1))
        
        # Parse "very big sphere" (scaled)
        hypotheses2 = latn_tokenize_layer1("very big sphere")
        tokens2 = hypotheses2[0].tokens  # Use best hypothesis
        result2 = run_np(TokenStream(tokens2))
        
        assert result1 is not None and result2 is not None
        
        # The "very big sphere" should have larger scale values than "big sphere"
        assert result2.vector["scaleX"] > result1.vector["scaleX"], "Very big should be bigger than just big"
        assert result2.vector["scaleY"] > result1.vector["scaleY"], "Very big should be bigger than just big"
        assert result2.vector["scaleZ"] > result1.vector["scaleZ"], "Very big should be bigger than just big"
    
    def test_traditional_determiner_nps_still_work(self):
        """Test that traditional determiner-initial NPs still work after enhancement."""
        test_cases = [
            ("the sphere", ["the", "sphere"]),
            ("a red box", ["a", "red", "box"]),
            ("the big blue cylinder", ["the", "big", "blue", "cylinder"])
        ]

        for case, expected_tokens in test_cases:
            hypotheses = latn_tokenize_layer1(case)
            tokens = hypotheses[0].tokens  # Use best hypothesis
            result = run_np(TokenStream(tokens))
            
            assert result is not None, f"Failed to parse traditional NP: '{case}'"
            assert result.get_consumed_words() == expected_tokens, f"Expected tokens {expected_tokens}, got {result.get_consumed_words()}"
            assert result.determiner is not None, f"Traditional NP should have determiner"
            
            # Check vector has determiner component
            assert result.vector["det"] > 0, f"Vector should have determiner component"
            assert result.vector["noun"] > 0, f"Vector should have noun component"
    
    def test_np_vector_composition(self):
        """Test that NP vectors properly combine tokens."""
        hypotheses = latn_tokenize_layer1("very big red sphere")
        tokens = hypotheses[0].tokens  # Use best hypothesis
        result = run_np(TokenStream(tokens))
        
        assert result is not None, "Failed to parse complex NP"
        assert result.get_consumed_words() == ["very", "big", "red", "sphere"]
        
        # Check that all components are present in the vector
        assert result.vector["noun"] > 0, "Should have noun component"
        assert result.vector["adj"] > 0, "Should have adjective component"  
        assert result.vector["red"] > 0, "Should have red color component"
        assert result.vector["scaleX"] > 2, "Should have scaled size component (very big)"
        assert result.vector["scaleY"] > 2, "Should have scaled size component (very big)"
        assert result.vector["scaleZ"] > 2, "Should have scaled size component (very big)"
    
    def test_semantic_attribute_extraction(self):
        """Test that NPs extract semantic attributes correctly for LATN Layer 2."""
        test_cases = [
            ("red sphere", {"red": True, "color_specified": True}),
            ("big sphere", {"size_specified": True, "scaled": True}),
            ("very tall cylinder", {"size_specified": True, "height_scaled": True}),
            ("small blue box", {"blue": True, "size_specified": True, "color_specified": True})
        ]

        for case, expected_attributes in test_cases:
            hypotheses = latn_tokenize_layer1(case)
            tokens = hypotheses[0].tokens  # Use best hypothesis
            result = run_np(TokenStream(tokens))
            
            assert result is not None, f"Failed to parse: '{case}'"
            
            # Check expected semantic attributes
            if expected_attributes.get("red"):
                assert result.vector["red"] > 0, f"'{case}' should have red component"
            if expected_attributes.get("blue"):
                assert result.vector["blue"] > 0, f"'{case}' should have blue component"
            if expected_attributes.get("size_specified"):
                has_scale = (result.vector["scaleX"] != 0 or 
                           result.vector["scaleY"] != 0 or 
                           result.vector["scaleZ"] != 0)
                assert has_scale, f"'{case}' should have scale components"
                
    def test_latn_hypothesis_readiness(self):
        """Test that enhanced NPs are ready for LATN multi-hypothesis generation."""
        # This tests that we can parse multiple NP patterns from the same token sequence
        # Use just the NP part without additional words that would confuse the parser
        hypotheses = latn_tokenize_layer1("very big red spheres")
        sentence_tokens = hypotheses[0].tokens  # Use best hypothesis
        
        # Test that we can find NPs starting at different positions
        np_hypotheses = []
        
        # Try parsing NP starting at position 0: "very big red spheres"
        tokens1 = TokenStream(sentence_tokens)
        result1 = run_np(tokens1)
        if result1:
            np_hypotheses.append((0, result1))
        
        # Try parsing NP starting at position 1: "big red spheres"  
        tokens2 = TokenStream(sentence_tokens[1:])
        result2 = run_np(tokens2)
        if result2:
            np_hypotheses.append((1, result2))
            
        # Try parsing NP starting at position 2: "red spheres"
        tokens3 = TokenStream(sentence_tokens[2:])
        result3 = run_np(tokens3)
        if result3:
            np_hypotheses.append((2, result3))
            
        # Try parsing NP starting at position 3: "spheres"
        tokens4 = TokenStream(sentence_tokens[3:])
        result4 = run_np(tokens4)
        if result4:
            np_hypotheses.append((3, result4))
        
        # Should find multiple valid NP hypotheses
        assert len(np_hypotheses) >= 2, f"Should find multiple NP hypotheses, found: {[(pos, np.get_consumed_words()) for pos, np in np_hypotheses]}"
        
        # Verify the hypotheses have different specificity levels
        hypothesis_lengths = [len(np.get_consumed_words()) for pos, np in np_hypotheses]
        assert len(set(hypothesis_lengths)) > 1, "Should have hypotheses of different lengths (specificity)"


