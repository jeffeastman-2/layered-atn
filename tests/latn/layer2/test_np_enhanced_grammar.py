"""Host-neutral tests for the noun-phrase ATN."""

import pytest

from latn.atn.subnet_np import run_np
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1
from latn.lexer.token_stream import TokenStream


pytestmark = pytest.mark.usefixtures("neutral_latn")


def parse_np(text):
    for hypothesis in latn_tokenize_layer1(text):
        result = run_np(TokenStream(hypothesis.tokens))
        if result:
            return result
    return None


@pytest.mark.parametrize("word", ["sphere", "box", "cylinder"])
def test_bare_noun_np(word):
    result = parse_np(word)
    assert result is not None
    assert result.noun == word
    assert result.determiner is None
    assert result.vector.isa("noun")


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("red sphere", ["red", "sphere"]),
        ("big box", ["big", "box"]),
        ("small cylinder", ["small", "cylinder"]),
        ("big red sphere", ["big", "red", "sphere"]),
        ("small blue box", ["small", "blue", "box"]),
    ],
)
def test_adjective_initial_nps(text, words):
    result = parse_np(text)
    assert result is not None
    assert result.get_consumed_words() == words
    assert result.vector.isa("adj")
    assert result.vector.isa("noun")


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("very big sphere", ["very", "big", "sphere"]),
        ("extremely red box", ["extremely", "red", "box"]),
        ("very small cylinder", ["very", "small", "cylinder"]),
    ],
)
def test_adverb_initial_nps(text, words):
    result = parse_np(text)
    assert result is not None
    assert result.get_consumed_words() == words
    assert result.vector["adj"] > 1.0


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("the sphere", ["the", "sphere"]),
        ("a red box", ["a", "red", "box"]),
        ("the big blue cylinder", ["the", "big", "blue", "cylinder"]),
    ],
)
def test_determiner_initial_nps(text, words):
    result = parse_np(text)
    assert result is not None
    assert result.get_consumed_words() == words
    assert result.vector.isa("det")


def test_host_semantics_are_transported_without_core_knowledge(neutral_latn):
    result = parse_np("very bright object")
    assert result is not None
    assert result.vector.isa("test_quality")
    assert result.vector["test_quality"] > neutral_latn["bright"]["test_quality"]


def test_multiple_np_start_positions_produce_hypotheses():
    tokens = latn_tokenize_layer1("very big red spheres")[0].tokens
    results = [run_np(TokenStream(tokens[start:])) for start in range(len(tokens))]
    assert sum(result is not None for result in results) >= 2
