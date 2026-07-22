"""Closed-class English vocabulary required by LATN's grammar.

Content words and semantic readings belong to host lexicons. Prepositions are
therefore marked only as prepositions here; a host decides what relations they
mean and whether those distinctions need executable dimensions.
"""

from latn.lexer.vector_space import vector_from_features


FUNCTION_WORDS = {
    # Pronouns and agreement.
    "it": vector_from_features("pronoun singular"),
    "they": vector_from_features("pronoun plural"),
    "them": vector_from_features("pronoun plural"),
    "i": vector_from_features("pronoun singular"),
    "me": vector_from_features("pronoun singular"),
    "you": vector_from_features("pronoun"),
    "we": vector_from_features("pronoun plural"),
    "us": vector_from_features("pronoun plural"),

    # Degree adverbs affect vector composition in the NP grammar.
    "very": vector_from_features("adv", adverb=1.5),
    "more": vector_from_features("adv", adverb=1.5),
    "much": vector_from_features("adv", adverb=1.5),
    "extremely": vector_from_features("adv", adverb=2.0),
    "slightly": vector_from_features("adv", adverb=0.75),

    # Determiners and grammatical number.
    "the": vector_from_features("det def"),
    "a": vector_from_features("det singular", number=1.0),
    "an": vector_from_features("det singular", number=1.0),
    **{
        word: vector_from_features(
            "det def singular" if value == 1 else "det def plural",
            number=float(value),
        )
        for value, word in enumerate(
            ("zero", "one", "two", "three", "four", "five", "six",
             "seven", "eight", "nine", "ten")
        )
        if value > 0
    },
    "my": vector_from_features("det"),
    "your": vector_from_features("det"),
    "his": vector_from_features("det"),
    "her": vector_from_features("det"),
    "its": vector_from_features("det"),
    "our": vector_from_features("det"),
    "their": vector_from_features("det"),

    # Quantifier-determiners exhibit determiner behavior in the NP grammar.
    **{
        word: vector_from_features("det")
        for word in ("some", "any", "all", "each", "every",
                     "several", "few", "both", "either", "neither")
    },

    # Relation words are grammar-only in the core.
    **{
        word: vector_from_features("prep")
        for word in (
            "over", "above", "under", "below", "behind", "in front of",
            "right of", "left of", "on", "in", "at", "near", "to",
            "toward", "towards", "into", "onto", "through", "from", "by",
            "with", "of", "as", "than", "around", "about", "for",
        )
    },
    **{
        word: vector_from_features("adv prep")
        for word in ("up", "down", "left", "right", "forward", "backward", "back")
    },

    "and": vector_from_features("conj and"),
    "or": vector_from_features("conj or"),
    "not": vector_from_features("neg"),
    "no": vector_from_features("neg"),
    **{
        word: vector_from_features("verb modal")
        for word in ("can", "could", "may", "might", "must", "shall", "should", "will", "would")
    },
    **{
        word: vector_from_features("wh")
        for word in ("who", "what", "where", "when", "why", "how", "which")
    },
    **{
        word: vector_from_features("tobe")
        for word in ("is", "are", "was", "were", "am", "be", "been", "being")
    },
    # Light and auxiliary verbs the grammar needs: do-support ("do you have"),
    # the perfect auxiliary "have", and the "would like to" frame. Grammar-only
    # here (POS, no reading) — a host supplies any semantic mapping, e.g.
    # Driftmoor grounds "have"/"want" to purchasing.
    **{
        word: vector_from_features("verb")
        for word in ("do", "does", "did", "have", "has", "had", "like")
    },
    ",": vector_from_features("punct comma"),
    ".": vector_from_features("punct period"),
    "?": vector_from_features("punct question"),
    "!": vector_from_features("punct exclaim"),
}


# Backward-compatible name for the core's active default table. It is no
# longer a semantic/CAD vocabulary; hosts supply their own Lexicon.
SEMANTIC_VECTOR_SPACE = FUNCTION_WORDS

__all__ = ["FUNCTION_WORDS", "SEMANTIC_VECTOR_SPACE"]
