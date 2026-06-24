import pytest
from latn.lexer.latn_layer_executor import tokenize_best
from latn.lexer.vector_space import VectorSpace
from latn.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from latn.lexer.vocabulary_builder import add_to_vocabulary, get_from_vocabulary, has_word

def assert_vector_has_pos(vs: VectorSpace, expected_pos: str):
    assert isinstance(vs, VectorSpace)
    for pos in expected_pos.split():
        assert pytest.approx(vs[pos], abs=1e-6) == 1.0, f"Expected POS '{pos}' not found in vector: {vs}"
        

def test_tokenize_simple_words():
    tokens = tokenize_best("draw a red cube")
    assert len(tokens) == 4
    assert_vector_has_pos(tokens[0], "verb")
    assert_vector_has_pos(tokens[1], "det")
    assert_vector_has_pos(tokens[2], "adj")
    assert_vector_has_pos(tokens[3], "noun")

def test_tokenize_vector_literal():
    tokens = tokenize_best("at [3.0, 4.0, 5.0]")
    assert len(tokens) == 2
    assert_vector_has_pos(tokens[0], "prep")
    assert_vector_has_pos(tokens[1], "vector")
    assert tokens[1]["locX"] == 3.0
    assert tokens[1]["locY"] == 4.0
    assert tokens[1]["locZ"] == 5.0

def test_tokenize_number():
    tokens = tokenize_best("draw 2 cube")
    assert len(tokens) == 3
    assert_vector_has_pos(tokens[0], "verb")
    assert_vector_has_pos(tokens[1], "det def")
    print(tokens[1])
    # Check if the number is correctly parsed as a float
    assert isinstance(tokens[1]["number"], float)
    assert tokens[1]["number"] == 2.0
    assert_vector_has_pos(tokens[2], "noun")

def test_tokenize_with_float_number():
    tokens = tokenize_best("draw 3.5 sphere")
    assert len(tokens) == 3
    assert_vector_has_pos(tokens[1], "det def")
    assert tokens[1]["number"] == 3.5

def test_tokenize_unknown_token_raises():
    # Unknown tokens now get marked with unknown=1.0 instead of raising errors
    tokens = list(tokenize_best("draw foozle"))
    assert len(tokens) == 2, "Should tokenize both known and unknown tokens"
    assert tokens[0].word == "draw", "First token should be 'draw'"
    assert tokens[1].word == "foozle", "Second token should be 'foozle'"
    assert tokens[1]["unknown"] == 1.0, "Unknown token should have unknown=1.0"

def test_tokenize_quoted_word():
    tokens = tokenize_best("'sky blue' is blue and green")
    assert len(tokens) == 5
    assert_vector_has_pos(tokens[0], "quoted")
    assert tokens[0].word == "sky blue"
    assert_vector_has_pos(tokens[1], "tobe")
    assert_vector_has_pos(tokens[2], "adj")
    assert_vector_has_pos(tokens[3], "conj")
    assert_vector_has_pos(tokens[4], "adj") 
    add_to_vocabulary(tokens[0].word, tokens[0])
    print(f"Added '{tokens[0].word}' to vocabulary: {SEMANTIC_VECTOR_SPACE[tokens[0].word]}")
    assert has_word("sky blue")

def test_tokenize_plural_words():
    tokens = tokenize_best("draw three red cubes")
    assert len(tokens) == 4
    assert_vector_has_pos(tokens[0], "verb")
    assert_vector_has_pos(tokens[1], "det")
    assert_vector_has_pos(tokens[2], "adj")
    assert_vector_has_pos(tokens[3], "noun")
    assert_vector_has_pos(tokens[3], "plural")

