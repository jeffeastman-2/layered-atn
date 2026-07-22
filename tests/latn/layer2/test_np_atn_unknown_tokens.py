#!/usr/bin/env python3
"""
Test for NP ATN handling of unknown tokens.

This test demonstrates the issue where unknown tokens can disrupt NP boundary detection,
and validates the fix to ensure proper phrase termination.
"""

import pytest
from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.atn.np import run_np
from latn.lexer.token_stream import TokenStream
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1
from latn.pos.noun_phrase import NounPhrase


pytestmark = pytest.mark.usefixtures("neutral_latn")


def test_unknown_token_after_noun_should_terminate_np():
    """Test that unknown token after noun properly terminates NP parsing."""
    
    # Get tokenized version of "box next" where "next" is unknown
    hypotheses = latn_tokenize_layer1("box next")
    tokens = hypotheses[0].tokens  # Get first hypothesis
    
    # Verify that "next" is marked as unknown
    box_token = tokens[0]
    next_token = tokens[1]
    assert box_token.word == "box"
    assert next_token.word == "next"
    assert box_token["unknown"] == 0.0, "box should be known"
    assert next_token["unknown"] == 1.0, "next should be unknown"
    
    # Test NP parsing - should only consume "box", not "box next"
    result = run_np(tokens)
    
    # Should successfully parse "box" as NP
    assert result is not None, "Should parse 'box' as valid NP"
    assert result.noun == "box", "Should extract 'box' as noun"
    
    # The key test: verify that only "box" was consumed, not "next"
    # We can test this by checking that the ATN would stop before "next"
    ts = TokenStream(tokens)
    from latn.atn.np import build_np_atn
    from latn.atn.core import run_atn
    
    np = NounPhrase()
    np_start, np_end = build_np_atn(np, ts)
    initial_pos = ts.position
    result = run_atn(np_start, np_end, ts, np)
    final_pos = ts.position
    tokens_consumed = final_pos - initial_pos
    
    assert tokens_consumed == 1, f"Should consume only 1 token (box), but consumed {tokens_consumed}"
    

def test_layer2_np_identification_with_unknown_tokens():
    """Test Layer 2 NP identification when unknown tokens disrupt parsing."""
    
    executor = LATNLayerExecutor()
    
    # Test case that currently fails: "a box next to a sphere"
    result_with_unknown = executor.execute_layer2("a box next to a sphere")
    
    # Test case that works: "a box above a sphere" 
    result_without_unknown = executor.execute_layer2("a box above a sphere")
    
    print(f"\nDEBUG: With unknown token 'next':")
    print(f"  Noun phrases: {len(result_with_unknown.noun_phrases)}")
    print(f"  Tokens: {[token.word for token in result_with_unknown.hypotheses[0].tokens]}")
    
    print(f"\nDEBUG: Without unknown token:")
    print(f"  Noun phrases: {len(result_without_unknown.noun_phrases)}")
    print(f"  Tokens: {[token.word for token in result_without_unknown.hypotheses[0].tokens]}")
    
    # The fix: unknown tokens now properly terminate NPs
    # New behavior: both NPs should be identified correctly
    # Expected behavior: 2 NPs should be identified in both cases
    
    # After implementing fix, both cases should identify 2 NPs
    assert len(result_with_unknown.noun_phrases) == 2, "Fixed behavior: unknown token properly terminates NP, both NPs identified"
    assert len(result_without_unknown.noun_phrases) == 2, "Control case: both NPs identified correctly"
    

def test_np_atn_should_terminate_on_unknown_after_noun():
    """Test that NP ATN properly terminates when encountering unknown token after noun."""
    
    # Create test case: determiner + noun + unknown
    hypotheses = latn_tokenize_layer1("a box next")
    tokens = hypotheses[0].tokens  # Get first hypothesis
    
    # Verify token properties
    a_token, box_token, next_token = tokens
    assert a_token.word == "a"
    assert box_token.word == "box" 
    assert next_token.word == "next"
    assert next_token["unknown"] == 1.0, "next should be unknown"
    
    # Parse with NP ATN
    ts = TokenStream(tokens)
    from latn.atn.np import build_np_atn
    from latn.atn.core import run_atn
    
    np = NounPhrase()
    np_start, np_end = build_np_atn(np, ts)
    initial_pos = ts.position
    result = run_atn(np_start, np_end, ts, np)
    final_pos = ts.position
    tokens_consumed = final_pos - initial_pos
    
    # Should consume "a box" (2 tokens) but stop at "next" (unknown)
    assert result is not None, "Should successfully parse 'a box'"
    assert tokens_consumed == 2, f"Should consume 'a box' (2 tokens), consumed {tokens_consumed}"
    assert result.determiner == "a"
    assert result.noun == "box"
