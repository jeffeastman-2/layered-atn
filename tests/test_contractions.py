"""Contraction expansion (latn.utils.contractions) + tokenizer integration.

The raw regex splits a contraction on its apostrophe, so the unit inputs here
are the post-split token lists the tokenizer actually sees (``I'd`` -> ["I","'","d"]).
"""

from latn.utils.contractions import expand_contractions
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_best
from latn.lexer.diagnostics import unknown_words


class TestExpandUnit:
    def test_unambiguous_clitics(self):
        assert expand_contractions(["I", "'", "m"]) == [["I", "am"]]
        assert expand_contractions(["you", "'", "re"]) == [["you", "are"]]
        assert expand_contractions(["I", "'", "ll"]) == [["I", "will"]]
        assert expand_contractions(["I", "'", "ve"]) == [["I", "have"]]

    def test_nt_regular_and_irregular(self):
        assert expand_contractions(["don", "'", "t"]) == [["do", "not"]]
        assert expand_contractions(["isn", "'", "t"]) == [["is", "not"]]
        assert expand_contractions(["can", "'", "t"]) == [["can", "not"]]
        assert expand_contractions(["won", "'", "t"]) == [["will", "not"]]

    def test_ambiguous_d_branches_would_and_had(self):
        assert expand_contractions(["I", "'", "d", "go"]) == [
            ["I", "would", "go"], ["I", "had", "go"]]

    def test_ambiguous_s_branches_is_has_possessive(self):
        # possessive variant drops the clitic (bare noun survives).
        assert expand_contractions(["it", "'", "s"]) == [
            ["it", "is"], ["it", "has"], ["it"]]

    def test_no_contraction_is_unchanged_single_variant(self):
        assert expand_contractions(["buy", "the", "bolts"]) == [["buy", "the", "bolts"]]

    def test_trailing_apostrophe_is_left_alone(self):
        # "dogs'" -> no tail after the quote; not a contraction.
        assert expand_contractions(["dogs", "'"]) == [["dogs", "'"]]


class TestTokenizerIntegration:
    def test_contraction_leaves_no_unknown_fragment(self):
        # 'd / 'm / n't must not survive as stray unknown tokens.
        for s in ("I'd go", "I'm here", "don't", "you're", "I've", "I'll", "won't"):
            words = [t.word for t in latn_tokenize_best(s)]
            assert "d" not in words and "m" not in words and "t" not in words, s
            assert "" not in words, s

    def test_would_reading_available_for_apostrophe_d(self):
        words = [t.word for t in latn_tokenize_best("I'd go")]
        assert "would" in words

    def test_diagnostic_does_not_flag_the_clitic(self):
        # The whole point: the diagnostic must not report 'd/'m/'s as unknowns.
        assert "d" not in unknown_words("I'd like it")
        assert "m" not in unknown_words("I'm here")
