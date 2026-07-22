"""Parser-honesty diagnostics (latn.lexer.diagnostics).

Runs against the default core lexicon (function words only), so the "known"
words here are grammatical ones -- enough to exercise detect + suggest without a
host vocabulary. Morphology-on-nouns is covered by the host suites.
"""

from latn.lexer.diagnostics import unknown_words, suggest_words, diagnose


class TestUnknownWords:
    def test_pure_function_word_line_has_no_unknowns(self):
        assert unknown_words("some of them") == []
        assert unknown_words("with each of the two") == []

    def test_flags_out_of_vocabulary(self):
        assert unknown_words("floober the grommit") == ["floober", "grommit"]

    def test_order_preserved_and_deduped(self):
        assert unknown_words("floober and floober and grommit") == ["floober", "grommit"]

    def test_known_pronoun_is_not_flagged(self):
        assert "them" not in unknown_words("give them to me")

    def test_empty_text(self):
        assert unknown_words("") == []


class TestSuggestions:
    def test_typo_of_a_known_word_is_suggested(self):
        # "teh" -> "the"; "somme" -> "some"; both are core function words.
        assert "the" in suggest_words("teh")
        assert "some" in suggest_words("somme")

    def test_nonsense_gets_no_suggestion(self):
        assert suggest_words("floober") == []

    def test_diagnose_pairs_unknown_with_suggestion(self):
        pairs = dict(diagnose("teh floober"))
        assert "the" in pairs["teh"]
        assert pairs["floober"] == []
