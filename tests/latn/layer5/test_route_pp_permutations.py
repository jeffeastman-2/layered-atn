from itertools import permutations

import pytest

from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.lexer.vector_space import vector_from_features
from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.pos.prepositional_phrase import PrepositionalPhrase
from latn.pos.sentence_phrase import SentencePhrase


ROUTE_PPS = (
    "over the mountains",
    "through the woods",
    "to Grandmother's house",
)
EXPECTED_ROUTE = {
    ("over", "mountain"),
    ("through", "wood"),
    ("to", "house"),
}


def _route_permutations():
    """All PP orders, split before/after the fixed subject-predicate clause."""
    cases = []
    for order in permutations(ROUTE_PPS):
        for split in range(len(order) + 1):
            parts = []
            if split:
                parts.append(" and ".join(order[:split]))
            parts.append("we go")
            if split < len(order):
                parts.append(" and ".join(order[split:]))
            sentence = " ".join(parts)
            cases.append(pytest.param(sentence, id=f"route-{len(cases) + 1:02d}"))
    return cases


def _flatten_conjunctions(phrase):
    if not isinstance(phrase, ConjunctionPhrase):
        return [phrase]
    flattened = []
    for part in phrase.phrases:
        flattened.extend(_flatten_conjunctions(part))
    return flattened


def _sentence_phrases(result):
    for hypothesis in result.hypotheses:
        for token in hypothesis.tokens:
            phrase = getattr(token, "phrase", None)
            if isinstance(phrase, SentencePhrase):
                yield phrase


def _route_from_sentence(sentence_phrase):
    route = []
    for phrase in getattr(sentence_phrase, "prepositional_phrases", []):
        route.extend(_flatten_conjunctions(phrase))
    if sentence_phrase.predicate is not None:
        for phrase in sentence_phrase.predicate.prepositions:
            route.extend(_flatten_conjunctions(phrase))
    return {
        (phrase.preposition.lower(), phrase.noun_phrase.noun)
        for phrase in route
        if isinstance(phrase, PrepositionalPhrase)
    }


@pytest.mark.parametrize("sentence", _route_permutations())
def test_route_pp_order_and_clause_position_are_semantically_equivalent(
    neutral_latn, sentence
):
    for word, features in {
        "over": "prep",
        "through": "prep",
        "mountain": "noun",
        "wood": "noun",
        "grandmother": "noun",
        "we": "pronoun plural",
        "go": "verb",
    }.items():
        neutral_latn[word] = vector_from_features(features)

    result = LATNLayerExecutor().execute_layer5(sentence, tokenize_only=True)

    assert result.success, result.description
    assert any(
        phrase.subject is not None
        and phrase.subject.pronoun == "we"
        and phrase.predicate is not None
        and phrase.predicate.verb == "go"
        and _route_from_sentence(phrase) == EXPECTED_ROUTE
        for phrase in _sentence_phrases(result)
    ), f"No equivalent route parse found for: {sentence}"

