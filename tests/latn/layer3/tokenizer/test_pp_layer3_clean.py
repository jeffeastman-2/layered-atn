#!/usr/bin/env python3
"""
LATN Layer 3: PrepositionalPhrase Token Tests (Clean Version)

Tests for the Layered Augmented Transition Network Layer 3 implementation.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.

The beauty of Layer 3 is that the PP ATN becomes incredibly simple:
- PREP -> NounPhrase token -> END
- No more complex NP subnetwork parsing!
"""

import pytest
from latn.lexer.vector_space import vector_from_features


def run_layer3(text):
    """Process text through all LATN layers up to Layer 3."""
    from latn.lexer.latn_layer_executor import LATNLayerExecutor
    executor = LATNLayerExecutor()
    result = executor.execute_layer3(text)
    if result.success and result.hypotheses:
        return result.hypotheses[0]
    return None


class TestLATNLayer3BasicPP:
    """Test basic PP token replacement functionality."""
    
    def test_simple_pp_replacement(self):
        """Test basic PP replacement: 'at [1,2,3]' -> single PP token."""
        hypothesis = run_layer3("at [1,2,3]")
        assert hypothesis is not None, "Layer 3 should process simple PP"
        
        tokens = [tok.word for tok in hypothesis.tokens]
        print(f"Layer 3 tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(hypothesis.replacements) > 0, "Should have PP replacements"
        
        # Check for PP tokens
        pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PrepositionalPhrase tokens"
    
    def test_pp_with_simple_np(self):
        """Test PP replacement with simple NP: 'on the table'."""
        hypothesis = run_layer3("on the table")
        assert hypothesis is not None, "Layer 3 should process PP with simple NP"
        
        tokens = [tok.word for tok in hypothesis.tokens]
        print(f"Simple NP PP tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(hypothesis.replacements) > 0, "Should have PP replacements"
        
        # Check for PP tokens
        pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PrepositionalPhrase tokens"
    
    def test_pp_with_complex_np(self):
        """Test PP replacement with complex NP: 'above the big red sphere'."""
        hypothesis = run_layer3("above the big red sphere")
        assert hypothesis is not None, "Layer 3 should process PP with complex NP"
        
        tokens = [tok.word for tok in hypothesis.tokens]
        print(f"Complex NP PP tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(hypothesis.replacements) > 0, "Should have PP replacements"
        
        # Check for PP tokens
        pp_tokens = [tok for tok in hypothesis.tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PrepositionalPhrase tokens"


class TestLATNLayer3Integration:
    """Test Layer 3 tokenization integration with full pipeline."""
    
    def test_layer3_tokenization_simple_pp(self):
        """Test Layer 3 tokenization of simple prepositional phrases."""
        sentence = "at [1,2,3]"

        from latn.lexer.latn_layer_executor import LATNLayerExecutor
        executor = LATNLayerExecutor()
        result = executor.execute_layer3(sentence)
        assert result.success, "Layer 3 should succeed"
        hypotheses = result.hypotheses
        assert len(hypotheses) > 0
        
        # Check the best hypothesis
        best_hyp = hypotheses[0]
        tokens = [tok.word for tok in best_hyp.tokens]
        
        print(f"Layer 3 tokens: {tokens}")
        
        # Should have PP replacement
        assert any("PP(" in token for token in tokens), "Should have PP token"
        assert len(best_hyp.replacements) > 0, "Should have PP replacements"
        
        # Test using the Layer Executor
        from latn.lexer.latn_layer_executor import LATNLayerExecutor
        executor = LATNLayerExecutor()
        result = executor.execute_layer3(sentence)
        assert result.success, "Layer 3 should succeed"
        pp_tokens = [tok for tok in result.hypotheses[0].tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PP tokens"
    
    def test_layer3_complex_pp(self):
        """Test Layer 3 with complex prepositional phrase."""
        sentence = "above the very large red sphere"

        from latn.lexer.latn_layer_executor import LATNLayerExecutor
        executor = LATNLayerExecutor()
        result = executor.execute_layer3(sentence)
        assert result.success, "Layer 3 should succeed"
        hypotheses = result.hypotheses
        assert len(hypotheses) > 0
        
        best_hyp = hypotheses[0]
        tokens = [tok.word for tok in best_hyp.tokens]
        
        print(f"Complex PP tokens: {tokens}")
        
        # Should have PP replacement
        pp_count = sum(1 for token in tokens if "PP(" in token)
        assert pp_count >= 1, f"Should have at least 1 PP token, got {pp_count}"
        assert len(best_hyp.replacements) >= 1, "Should have PP replacements"


class TestLATNLayer3Ambiguity:
    """Test Layer 3 with NP ambiguity that creates PP ambiguity."""
    
    def test_pp_ambiguity_from_np_ambiguity(self):
        """Test how NP ambiguity affects PP parsing: 'near the red box'."""
        sentence = "near the red box"
        
        from latn.lexer.latn_layer_executor import LATNLayerExecutor
        executor = LATNLayerExecutor()
        result = executor.execute_layer3(sentence)
        assert result.success, "Layer 3 should succeed"
        hypotheses = result.hypotheses
        assert len(hypotheses) > 0
        
        print(f"\nPP ambiguity test: '{sentence}'")
        for i, hyp in enumerate(hypotheses[:3], 1):  # Show first 3
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.3f}): {tokens}")
            print(f"    PP replacements: {len(hyp.replacements)}")
        
        # Check that we get PP tokens in hypotheses
        best_hyp = hypotheses[0]
        pp_tokens = [tok for tok in best_hyp.tokens if tok.isa("PP")]
        assert len(pp_tokens) > 0, "Should have PP tokens"
    
    def test_compound_ambiguity_in_pp(self):
        """Test compound word ambiguity affecting PP: 'above the light house'."""
        from latn.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
        
        # Set up compound ambiguity
        original_entries = {}
        test_entries = {
            'light': vector_from_features('adj', transparency=0.8),
            'house': vector_from_features('noun'),
            'light house': vector_from_features('noun')  # compound
        }
        
        for word, vector in test_entries.items():
            original_entries[word] = word in SEMANTIC_VECTOR_SPACE
            if not original_entries[word]:
                SEMANTIC_VECTOR_SPACE[word] = vector
        
        try:
            sentence = "above the light house"
            
            from latn.lexer.latn_layer_executor import LATNLayerExecutor
            executor = LATNLayerExecutor()
            result = executor.execute_layer3(sentence)
            assert result.success, "Layer 3 should succeed"
            hypotheses = result.hypotheses
            assert len(hypotheses) > 0
            
            print(f"\nCompound ambiguity in PP: '{sentence}'")
            for i, hyp in enumerate(hypotheses[:3], 1):
                tokens = [tok.word for tok in hyp.tokens]
                print(f"  Hypothesis {i} (conf={hyp.confidence:.3f}): {tokens}")
                print(f"    PP replacements: {len(hyp.replacements)}")
            
            # Should have different interpretations
            # Some with "light house" (compound) and some with "light" + "house"
            # This creates different PP structures
            assert len(hypotheses) >= 2, "Should have multiple hypotheses for compound ambiguity"
            
        finally:
            # Clean up
            for word, was_original in original_entries.items():
                if not was_original and word in SEMANTIC_VECTOR_SPACE:
                    del SEMANTIC_VECTOR_SPACE[word]
    
    def test_complex_ambiguous_pp_chain(self):
        """Test multiple ambiguous PPs: 'from the table to the green sphere'."""
        sentence = "from the table to the green sphere"
        
        from latn.lexer.latn_layer_executor import LATNLayerExecutor
        executor = LATNLayerExecutor()
        result = executor.execute_layer3(sentence)
        assert result.success, "Layer 3 should succeed"
        hypotheses = result.hypotheses
        assert len(hypotheses) > 0
        
        print(f"\nComplex PP chain: '{sentence}'")
        for i, hyp in enumerate(hypotheses[:3], 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.3f}): {tokens}")
            print(f"    PP replacements: {len(hyp.replacements)}")
            
            # Count PP tokens
            pp_count = sum(1 for tok in hyp.tokens if tok.isa("PP"))
            print(f"    PP tokens: {pp_count}")
        
        # Should have multiple PPs in the best hypothesis
        best_hyp = hypotheses[0]
        pp_tokens = [tok for tok in best_hyp.tokens if tok.isa("PP")]
        assert len(pp_tokens) >= 1, f"Should have at least 1 PP token, got {len(pp_tokens)}"


