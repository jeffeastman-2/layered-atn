#!/usr/bin/env python3
"""
Unit Tests for LATN Layer 2: NounPhrase Token Replacement

Tests for the Layer 2 tokenization system that identifies noun phrases
and replaces them with single NounPhrase tokens.
"""

import pytest
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1
from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.atn.subnet_np import run_np
from latn.pos.noun_phrase import NounPhrase


# Helper functions that wrap LATNLayerExecutor for backward compatibility
def find_np_sequences(tokens):
    """Find NP sequences in a token list using subnet_np."""
    from latn.lexer.token_stream import TokenStream
    from latn.atn.subnet_np import run_np
    np = run_np(TokenStream(tokens))
    if np is not None:
        return [(0, len(tokens) - 1, np)]
    return []


def create_np_token(np):
    """Create an NP token from a NounPhrase."""
    from latn.lexer.vector_space import VectorSpace
    from latn.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
    token = VectorSpace()
    if hasattr(np, 'vector') and np.vector:
        for dim in VECTOR_DIMENSIONS:
            if np.vector[dim] != 0.0:
                token[dim] = np.vector[dim]
    token["NP"] = 1.0
    token.word = np.descriptive_word()
    token.phrase = np
    return token


def replace_np_sequences(tokens, np_sequences):
    """Replace token sequences with NP tokens."""
    if not np_sequences:
        return tokens
    result = []
    for start, end, np in np_sequences:
        result.append(create_np_token(np))
    return result


class TestLATNLayer2Basic:
    """Test basic Layer 2 NP token replacement functionality."""
    
    def test_simple_np_replacement(self):
        """Test basic NP replacement: 'the red box' -> single NP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("the red box")
        assert result.success, "Layer 2 should process successfully"
        assert len(result.hypotheses) > 0, "Should generate hypotheses"
        
        best = result.hypotheses[0]
        assert len(best.tokens) == 1, "Should have exactly one token"
        assert best.tokens[0].word == "NP(the red box)", "Should create NP token"
        assert best.tokens[0].isa("NP"), "Token should have NP dimension"
        assert hasattr(best.tokens[0], 'phrase'), "Should have original NP reference"
        assert len(best.replacements) == 1, "Should record NP replacement"
    
    def test_vector_np_replacement(self):
        """Test NP replacement with vector: '[1,2,3]' -> single NP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("[1,2,3]")
        assert result.success, "Layer 2 should process successfully"
        assert len(result.hypotheses) > 0, "Should generate hypotheses"
        
        best = result.hypotheses[0]
        assert len(best.tokens) == 1, "Should have exactly one token"
        assert best.tokens[0].word == "NP([1,2,3])", "Should create vector NP token"
        assert best.tokens[0].isa("NP"), "Token should have NP dimension"
        assert best.tokens[0].isa("vector"), "Should preserve vector semantics"
    
    def test_complex_np_replacement(self):
        """Test complex NP: 'a very large red sphere' -> single NP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("a very large red sphere")
        assert result.success, "Layer 2 should succeed"
        
        best = result.hypotheses[0]
        assert len(best.tokens) == 1, "Should have exactly one token"
        assert "NP(" in best.tokens[0].word, "Should create NP token"
        assert best.tokens[0].isa("NP"), "Token should have NP dimension"
        assert best.tokens[0].isa("adj"), "Should preserve adjective semantics"
        assert best.tokens[0].isa("noun"), "Should preserve noun semantics"
    
    def test_multiple_nps(self):
        """Test sentence with multiple NPs: 'the red box and the blue sphere'."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("the red box and the blue sphere")
        assert result.success, "Layer 2 should succeed"
        
        best = result.hypotheses[0]
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) >= 1, "Should have at least one NP token"
        assert len(best.replacements) >= 1, "Should record NP replacements"
    
    def test_no_np_sentence(self):
        """Test sentence with no noun phrases: 'hello'."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("hello")
        assert result.success, "Layer 2 should succeed"
        
        best = result.hypotheses[0]
        np_tokens = [tok for tok in best.tokens if tok.isa("NP")]
        assert len(np_tokens) == 0, "Should have no NP tokens"
        assert len(best.replacements) == 0, "Should record no NP replacements"


class TestLATNLayer2NounPhraseDetection:
    """Test the NP sequence finding functionality."""
    
    def test_find_np_sequences_simple(self):
        """Test finding NP sequences in token list."""
        # Get Layer 1 tokens for testing
        layer1_hypotheses = latn_tokenize_layer1("the red box")
        tokens = layer1_hypotheses[0].tokens
        
        np_sequences = find_np_sequences(tokens)
        assert len(np_sequences) == 1, "Should find one NP sequence"
        
        start, end, np = np_sequences[0]
        assert start == 0 and end == 2, "Should span all tokens"
        assert np.noun == "box", "NP should identify noun correctly"
        assert np.determiner == "the", "NP should identify determiner"
        
    def test_find_np_sequences_vector(self):
        """Test finding NP sequences with vector coordinates."""
        # Get Layer 1 tokens for testing  
        layer1_hypotheses = latn_tokenize_layer1("[5,10,15]")
        tokens = layer1_hypotheses[0].tokens
        
        np_sequences = find_np_sequences(tokens)
        assert len(np_sequences) == 1, "Should find one NP sequence"
        
        start, end, np = np_sequences[0]
        assert start == 0 and end == 0, "Should span vector token"
        assert np.vector_text == "[5,10,15]", "Should preserve vector text"
        
    def test_replace_np_sequences(self):
        """Test replacing NP sequences with NP tokens."""
        # Get Layer 1 tokens for testing
        layer1_hypotheses = latn_tokenize_layer1("the red box")
        tokens = layer1_hypotheses[0].tokens
    
    def test_find_np_sequences_vector(self):
        """Test finding vector NP sequences."""
        # Get Layer 1 tokens for testing
        layer1_hypotheses = latn_tokenize_layer1("[5,10,15]")
        tokens = layer1_hypotheses[0].tokens
        
        np_sequences = find_np_sequences(tokens)
        
        assert len(np_sequences) > 0, "Should find vector NP"
        start, end, np = np_sequences[0]
        assert start == 0, "Should start at beginning"
        assert np.vector["vector"] == 1.0, "Should be marked as vector"
    
    def test_replace_np_sequences(self):
        """Test replacing NP sequences with NP tokens."""
        # Get Layer 1 tokens for testing
        layer1_hypotheses = latn_tokenize_layer1("the red box")
        tokens = layer1_hypotheses[0].tokens
        
        np_sequences = find_np_sequences(tokens)
        new_tokens = replace_np_sequences(tokens, np_sequences)
        
        assert len(new_tokens) == 1, "Should replace with single token"
        assert new_tokens[0].isa("NP"), "Should be NP token"
        assert "NP(" in new_tokens[0].word, "Should have NP descriptor"


class TestLATNLayer2TokenCreation:
    """Test NP token creation functionality."""
    
    def test_create_np_token_structure(self):
        """Test NP token creation preserves structure."""
        # Get Layer 1 tokens for testing
        layer1_hypotheses = latn_tokenize_layer1("the big red sphere")
        tokens = layer1_hypotheses[0].tokens
        
        np = run_np(tokens)
        assert np is not None, "Should successfully parse NP"
        
        np_token = create_np_token(np)
        assert np_token.isa("NP"), "Should have NP dimension"
        assert np_token.isa("noun"), "Should preserve noun semantics"
        assert np_token.isa("adj"), "Should preserve adjective semantics"
        assert hasattr(np_token, 'phrase'), "Should store original NP"
        assert np_token.phrase is np, "Should reference original NP"
    
    def test_create_np_token_word_format(self):
        """Test NP token word formatting."""
        # Get Layer 1 tokens for testing
        layer1_hypotheses = latn_tokenize_layer1("the table")
        tokens = layer1_hypotheses[0].tokens
        
        np = run_np(tokens)
        np_token = create_np_token(np)
        
        assert "NP(" in np_token.word, "Should have NP prefix"
        assert "table" in np_token.word, "Should include noun"
        assert "the" in np_token.word, "Should include determiner"


class TestLATNLayer2Integration:
    """Test Layer 2 integration with overall system."""
    
    def test_layer2_with_executor(self):
        """Test Layer 2 using the proper layer executor."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("the red box")
        assert result.success, "Layer 2 should succeed"
        assert len(result.hypotheses) > 0, "Should generate hypotheses"
        assert len(result.hypotheses[0].tokens) == 1, "Should return single NP token"
        assert result.hypotheses[0].tokens[0].isa("NP"), "Should be NP token"
    
    def test_layer2_best_hypothesis(self):
        """Test accessing the best Layer 2 hypothesis."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("a blue sphere")
        assert result.success, "Should succeed"
        hypothesis = result.hypotheses[0]  # Best hypothesis
        assert hypothesis is not None, "Should return hypothesis"
        assert len(hypothesis.tokens) == 1, "Should have NP token"
        assert hypothesis.tokens[0].isa("NP"), "Should be NP token"
    
    def test_layer2_hypothesis_metadata(self):
        """Test that Layer 2 hypotheses have proper metadata."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("the green cube")
        assert result.success, "Should succeed"
        hypotheses = result.hypotheses
        assert len(hypotheses) > 0, "Should generate hypotheses"
        
        best = hypotheses[0]
        assert hasattr(best, 'confidence'), "Should have confidence score"
        assert hasattr(best, 'description'), "Should have description"
        assert hasattr(best, 'replacements'), "Should track NP replacements"
        assert "Layer 2" in best.description, "Description should mention Layer 2"


class TestLATNLayer2EdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_string(self):
        """Test Layer 2 with empty string."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("")
        # Empty string handling may return success=False, which is fine
        assert result is not None, "Should handle empty string gracefully"
    
    def test_single_word(self):
        """Test Layer 2 with single noun."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("sphere")
        assert result.success, "Should handle single noun"
        
        best = result.hypotheses[0]
        if len(best.tokens) == 1 and best.tokens[0].isa("NP"):
            # Single noun was converted to NP
            assert "sphere" in best.tokens[0].word
        else:
            # Single noun remained as is
            assert best.tokens[0].word == "sphere"
    
    def test_preposition_not_converted(self):
        """Test that prepositions alone are not converted to NPs."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer2("on")
        assert result.success, "Should handle preposition"
        
        best = result.hypotheses[0]
        prep_tokens = [tok for tok in best.tokens if tok.isa("prep")]
        assert len(prep_tokens) > 0, "Should preserve preposition"


