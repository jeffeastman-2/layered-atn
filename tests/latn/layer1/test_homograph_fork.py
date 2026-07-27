"""Homograph forking in the layer-1 tokenizer.

A single surface form that is a homograph across distinct parts of speech in the
active lexicon -- "leaves" as the noun "leaf" and the verb "leave" -- yields one
tokenization hypothesis per reading, so the grammar can keep whichever parses.
A plain plural (noun<->its own singular) or a dual-typed single entry does NOT
fork -- only genuinely distinct POS readings do, which keeps the beam bounded.
"""

from latn.lexer.lexicon import Lexicon, use_lexicon
from latn.lexer.vp_policy import use_vp_policy
from latn.lexer.sp_policy import use_sp_policy
from latn.lexer.vector_space import vector_from_features as vf
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1
from latn.lexer.latn_layer_executor import LATNLayerExecutor

from tests.latn.support import TestVPPolicy, TestSPPolicy


def _pos_of_single(word, table):
    """The (word, pos) of the first token of each hypothesis for a single word."""
    with use_lexicon(Lexicon(table)):
        hyps = latn_tokenize_layer1(word)
    tags = []
    for h in hyps:
        t = h.tokens[0]
        pos = next((p for p in ("noun", "verb", "adj") if t.isa(p)), "?")
        tags.append((t.word, pos))
    return tags


def test_homograph_forks_into_both_pos():
    tags = _pos_of_single("leaves", {"leaf": vf("noun singular"),
                                     "leaves": vf("verb")})
    pos = {p for _w, p in tags}
    assert "noun" in pos and "verb" in pos, tags


def test_single_reading_when_no_ambiguity():
    # Only the verb entry exists -> one reading, no spurious fork.
    tags = _pos_of_single("leaves", {"leaves": vf("verb")})
    assert {p for _w, p in tags} == {"verb"}, tags


def test_plain_plural_does_not_fork():
    # box/boxes are the same POS -> no homograph, one noun reading.
    tags = _pos_of_single("boxes", {"box": vf("noun singular")})
    assert {p for _w, p in tags} == {"noun"}, tags


def test_dual_typed_single_entry_does_not_fork():
    # "rocky" carries noun AND adj in ONE entry -> one token, not two hypotheses.
    with use_lexicon(Lexicon({"rocky": vf("noun adj")})):
        hyps = latn_tokenize_layer1("rocky")
    assert len(hyps) == 1
    assert hyps[0].tokens[0].isa("noun") and hyps[0].tokens[0].isa("adj")


def _executor_lexicon():
    return Lexicon({
        "the": vf("det def"), "leaf": vf("noun singular"),
        "leaves": vf("verb"), "she": vf("pronoun singular"),
        "fell": vf("verb"),
    })


def test_the_leaves_parses_as_noun_phrase():
    # After a determiner, the noun reading of "leaves" builds an NP.
    ex = LATNLayerExecutor()
    with use_lexicon(_executor_lexicon()), use_vp_policy(TestVPPolicy()), \
            use_sp_policy(TestSPPolicy()):
        result = ex.execute_layer2("the leaves", tokenize_only=True)
    assert any(h.tokens and h.tokens[0].isa("NP") for h in result.hypotheses)


def test_she_leaves_parses_as_verb_phrase():
    # With a subject pronoun, the verb reading of "leaves" builds a VP.
    ex = LATNLayerExecutor()
    with use_lexicon(_executor_lexicon()), use_vp_policy(TestVPPolicy()), \
            use_sp_policy(TestSPPolicy()):
        result = ex.execute_layer5("she leaves", tokenize_only=True)
    assert any(t.isa("VP")
               for h in result.layer4_result.hypotheses for t in h.tokens)
