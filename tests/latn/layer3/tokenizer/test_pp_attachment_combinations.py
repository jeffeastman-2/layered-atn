from latn.lexer.hypothesis import TokenizationHypothesis
from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.lexer.vector_space import VectorSpace
from latn.pos.noun_phrase import NounPhrase
from latn.pos.prepositional_phrase import PrepositionalPhrase


def _phrase_token(phrase_type, phrase):
    token = VectorSpace()
    token[phrase_type] = 1.0
    token.phrase = phrase
    token.word = phrase_type
    return token


def _pp(preposition, noun):
    phrase = PrepositionalPhrase()
    phrase.preposition = preposition
    phrase.noun_phrase = NounPhrase(noun)
    return phrase


def test_hypothesis_without_pp_is_preserved():
    hypothesis = TokenizationHypothesis(
        tokens=[_phrase_token("NP", NounPhrase("bells"))],
        confidence=1.0,
        description="no PP",
    )

    combinations = LATNLayerExecutor._generate_pp_attachment_combinations(
        [hypothesis]
    )

    assert combinations == [hypothesis]
    assert combinations[0] is hypothesis


def test_each_pp_can_remain_free_or_attach_to_a_preceding_np_or_pp():
    hypothesis = TokenizationHypothesis(
        tokens=[
            _phrase_token("NP", NounPhrase("we")),
            _phrase_token("PP", _pp("over", "mountains")),
            _phrase_token("PP", _pp("through", "woods")),
        ],
        confidence=1.0,
        description="two PPs",
    )

    combinations = LATNLayerExecutor._generate_pp_attachment_combinations(
        [hypothesis]
    )

    # The first PP has two choices; the second has three.
    assert len(combinations) == 6
    assert [len(item.tokens) for item in combinations] == [3, 2, 2, 2, 1, 1]

    pp_to_pp = next(
        item
        for item in combinations
        if len(item.tokens) == 2
        and item.tokens[1].isa("PP")
        and item.tokens[1].phrase.noun_phrase.prepositions
    )
    attached = pp_to_pp.tokens[1].phrase.noun_phrase.prepositions
    assert [phrase.preposition for phrase in attached] == ["through"]

    # Expansion must not mutate the tokenizer's source hypothesis.
    assert hypothesis.tokens[0].phrase.prepositions == []
    assert hypothesis.tokens[1].phrase.noun_phrase.prepositions == []


def test_invalid_attachment_branches_are_pruned_before_completion():
    hypothesis = TokenizationHypothesis(
        tokens=[
            _phrase_token("NP", NounPhrase("we")),
            _phrase_token("PP", _pp("over", "mountains")),
            _phrase_token("PP", _pp("through", "woods")),
        ],
        confidence=1.0,
        description="two PPs",
    )
    completed = []

    combinations = LATNLayerExecutor._generate_pp_attachment_combinations(
        [hypothesis],
        attachment_validator=lambda target: False,
        hypothesis_validator=lambda item: completed.append(item) or True,
    )

    # With no pruning this search has six completed combinations. Rejecting an
    # attachment immediately leaves only the structurally valid all-free path.
    assert len(completed) == 1
    assert combinations == completed
    assert len(combinations[0].tokens) == 3
