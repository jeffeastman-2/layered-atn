"""The host-neutral LATN vector schema.

Every built-in dimension below changes tokenization, an ATN transition,
agreement, or phrase construction. Hosts append executable semantic axes with
``register_dimensions`` before constructing vectors.
"""

VECTOR_DIMENSIONS = [
    # Lexical grammar.
    "verb", "tobe", "prep", "det", "def", "adv", "adj", "noun",
    "proper_noun", "pronoun", "unknown",

    # Composite tokens produced by the layered parser.
    "NP", "PP", "VP", "SP",

    # Agreement and grammatical operators.
    "number", "singular", "plural", "conj", "or", "and", "neg",
    "modal", "wh", "measure", "literal",

    # Morphology.
    "verb_past", "verb_past_part", "verb_present_part", "gerund",
    "comp", "super",

    # Token forms and parser punctuation.
    "quoted", "punct", "comma", "period", "exclaim", "question",
]


POS_DIMENSIONS = set(VECTOR_DIMENSIONS)
SEMANTIC_DIMENSIONS = set()


def get_semantic_mask():
    return [dim in SEMANTIC_DIMENSIONS for dim in VECTOR_DIMENSIONS]


def get_pos_mask():
    return [dim in POS_DIMENSIONS for dim in VECTOR_DIMENSIONS]


def register_dimensions(names, *, semantic=True):
    """Append host dimensions without changing existing vector indices."""
    for name in names:
        if name in VECTOR_DIMENSIONS:
            continue
        VECTOR_DIMENSIONS.append(name)
        (SEMANTIC_DIMENSIONS if semantic else POS_DIMENSIONS).add(name)

    from latn.lexer import vector_space as _vs
    _vs.VECTOR_LENGTH = len(VECTOR_DIMENSIONS)
    return len(VECTOR_DIMENSIONS)


__all__ = [
    "VECTOR_DIMENSIONS", "POS_DIMENSIONS", "SEMANTIC_DIMENSIONS",
    "get_semantic_mask", "get_pos_mask", "register_dimensions",
]
