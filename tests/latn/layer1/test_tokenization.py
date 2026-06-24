#!/usr/bin/env python3
"""
LATN Layer 1: Multi-Hypothesis Tokenization Tests

Tests for:
- Multi-hypothesis tokenization 
- Morphological inflection handling
- TokenizationHypothesis generation
- Confidence scoring for tokenization
"""

from latn.lexer.latn_layer_executor import tokenize_all, tokenize_best


def test_latn_layer1_morphological_inflection():
    """Test that LATN Layer 1 handles morphological inflections correctly."""
    
    # Test singular vs plural nouns
    singular_result = tokenize_all("sphere")
    plural_result = tokenize_all("spheres")
    
    # Should find tokenization for both
    assert len(singular_result) > 0
    assert len(plural_result) > 0
    
    # Singular should not have plural marking
    singular_tokens = singular_result[0].tokens
    sphere_tok = [t for t in singular_tokens if t.word == "sphere"][0]
    assert sphere_tok["noun"] > 0.0
    assert sphere_tok["plural"] == 0.0
    
    # Plural should have plural marking
    plural_tokens = plural_result[0].tokens  
    sphere_plural_tok = [t for t in plural_tokens if t.word == "sphere"][0]
    assert sphere_plural_tok["noun"] > 0.0
    assert sphere_plural_tok["plural"] > 0.0


def test_latn_layer1_verb_inflection():
    """Test LATN Layer 1 verb inflection handling."""
    
    # Test past participle
    past_part_result = tokenize_all("called")
    assert len(past_part_result) > 0
    tokens = past_part_result[0].tokens
    assert len(tokens) == 1
    tok = tokens[0]
    
    assert tok.word == "called"
    assert tok["verb"] > 0.0
    assert tok["verb_past_part"] > 0.0
    assert tok["action"] > 0.0
    assert tok["naming"] > 0.0
    
    # Test present participle  
    present_part_result = tokenize_all("calling")
    assert len(present_part_result) > 0
    tokens = present_part_result[0].tokens
    assert len(tokens) == 1
    tok = tokens[0]
    
    assert tok.word == "calling"
    assert tok["verb"] > 0.0
    assert tok["verb_present_part"] > 0.0
    assert tok["action"] > 0.0
    assert tok["naming"] > 0.0


def test_latn_layer1_multi_hypothesis_generation():
    """Test that LATN Layer 1 generates multiple tokenization hypotheses."""
    
    # Simple case should still generate at least one hypothesis
    result = tokenize_all("red sphere")
    assert len(result) > 0
    
    # Each hypothesis should have confidence and tokens
    for hyp in result:
        assert hasattr(hyp, 'confidence')
        assert hasattr(hyp, 'tokens')
        assert hyp.confidence > 0.0
        assert len(hyp.tokens) > 0


def test_latn_layer1_best_tokenization():
    """Test that LATN Layer 1 can select best tokenization hypothesis."""
    
    sentence = "big red spheres"
    best_tokens = tokenize_best(sentence)
    
    # Should get tokens for each word
    assert len(best_tokens) == 3
    
    # Check that we get the expected tokens
    words = [tok.word for tok in best_tokens]
    assert "big" in words
    assert "red" in words  
    assert "sphere" in words  # Should be singularized from "spheres"
    
    # The sphere token should have plural marking from inflection
    sphere_tok = [t for t in best_tokens if t.word == "sphere"][0]
    assert sphere_tok["plural"] > 0.0


def test_latn_layer1_adjective_inflection():
    """Test LATN Layer 1 adjective inflection handling."""
    
    # Test comparative and superlative forms would go here
    # For now, test basic adjective tokenization
    result = tokenize_all("bigger")
    assert len(result) > 0
    
    # Should handle "bigger" -> "big" + comparative marking
    tokens = result[0].tokens
    if len(tokens) == 1:  # If we have comparative inflection
        tok = tokens[0]
        # Either "bigger" as-is or "big" with comparative marking
        assert tok.word in ["bigger", "big"]
        assert tok["adj"] > 0.0


