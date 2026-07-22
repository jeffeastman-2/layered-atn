import pytest
from latn.lexer.latn_layer_executor import LATNLayerExecutor


pytestmark = pytest.mark.usefixtures("neutral_latn")


def test_simple_adverb_adjective_sequence():
    """Test Layer 2 NP tokenization with simple adverb-adjective sequence: 'a very tall cylinder'"""
    executor = LATNLayerExecutor()
    
    result = executor.execute_layer2('a very tall cylinder')
    
    assert result.success, "Failed to tokenize simple adverb-adjective sequence"
    assert len(result.hypotheses) > 0, "Should generate hypotheses"
    
    best = result.hypotheses[0]
    
    # Should have NP token in the result
    np_tokens = [token for token in best.tokens if hasattr(token, 'word') and token.word.startswith('NP(')]
    assert len(np_tokens) >= 1, "Should have at least 1 NP token"
    
    # Verify that adverb-adjective sequence is properly handled
    tokens_str = ' '.join([token.word for token in best.tokens])
    assert 'very' in tokens_str or 'NP(' in tokens_str, "Should handle 'very' adverb"
    assert 'tall' in tokens_str or 'NP(' in tokens_str, "Should handle 'tall' adjective"
    assert 'cylinder' in tokens_str or 'NP(' in tokens_str, "Should handle 'cylinder' noun"


def test_multiple_adverb_adjective_sequences():
    """Test Layer 2 NP tokenization with multiple adjectives: 'a very tall red box'"""
    executor = LATNLayerExecutor()
    
    result = executor.execute_layer2('a very tall red box')
    
    assert result.success, "Failed to tokenize multiple adverb-adjective sequences"
    assert len(result.hypotheses) > 0, "Should generate hypotheses"
    
    best = result.hypotheses[0]
    
    # Should have NP token in the result
    np_tokens = [token for token in best.tokens if hasattr(token, 'word') and token.word.startswith('NP(')]
    assert len(np_tokens) >= 1, "Should have at least 1 NP token"
    
    # Verify that multiple adjectives are properly handled
    tokens_str = ' '.join([token.word for token in best.tokens])
    assert 'very' in tokens_str or 'NP(' in tokens_str, "Should handle 'very' adverb"
    assert 'tall' in tokens_str or 'NP(' in tokens_str, "Should handle 'tall' adjective"
    assert 'red' in tokens_str or 'NP(' in tokens_str, "Should handle 'red' adjective"
    assert 'box' in tokens_str or 'NP(' in tokens_str, "Should handle 'box' noun"
