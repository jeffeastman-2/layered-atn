"""Injectable lexicon: the word -> VectorSpace table the LATN tokenizer uses.

Phase-1 seam 2 of factoring the LATN core out of Engraf. The grammar/ATN is
domain-general; the only domain-specific inputs are the lexicon and the
dimension schema. This module makes the lexicon swappable so Driftmoor can
supply an RPG vocabulary without forking the parser.

Behavior-preserving by construction: the default active lexicon wraps the
SAME latn.An_N_Space_Model.vocabulary.SEMANTIC_VECTOR_SPACE dict object by
reference, so for Engraf every lookup/mutation is byte-identical to before.

English morphology (singularize/verb+adjective inflection) deliberately
stays in the core (vocabulary_builder / latn_tokenizer_layer1 / verb_inflector)
and is NOT part of the injected table -- only the domain word table is
swapped; morphology is language-general.
"""

from typing import Iterator, MutableMapping, Optional

from latn.lexer.vector_space import VectorSpace


class Lexicon:
    """A word -> VectorSpace table behind a dict-style facade.

    Implements exactly the operations the core performs on the old global
    (`word in lex`, `lex[word]`, `lex.get(word)`, `lex[word] = vs`) so the
    rewire of existing call sites is mechanical and semantics-preserving.
    Backed by a caller-supplied dict, shared by reference (not copied).
    """

    def __init__(self, table: MutableMapping[str, VectorSpace]):
        self._table = table

    def __contains__(self, word: object) -> bool:
        return word in self._table

    def __getitem__(self, word: str) -> VectorSpace:
        return self._table[word]

    def __setitem__(self, word: str, vs: VectorSpace) -> None:
        self._table[word] = vs

    def get(self, word: str, default: Optional[VectorSpace] = None):
        return self._table.get(word, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)


def _build_default_lexicon() -> Lexicon:
    # Imported here (not at module top) to keep the import graph acyclic:
    # vocabulary imports vector_space; nothing in that chain imports lexicon.
    from latn.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE

    # Wrap the existing dict BY REFERENCE: add_to_vocabulary mutations and any
    # remaining direct readers of the global stay consistent with this lexicon.
    return Lexicon(SEMANTIC_VECTOR_SPACE)


_active: Optional[Lexicon] = None


def get_active_lexicon() -> Lexicon:
    """The lexicon the tokenizer/morphology consult. Call at use-time (do not
    bind at import) so set_active_lexicon / use_lexicon take effect."""
    global _active
    if _active is None:
        _active = _build_default_lexicon()
    return _active


def set_active_lexicon(lexicon: Optional[Lexicon]) -> None:
    """Swap the active lexicon. Pass None to reset to core function words."""
    global _active
    _active = lexicon


class use_lexicon:
    """Context manager: activate `lexicon` for a scope, restore the prior one.

    Intended for per-parse activation (Driftmoor) and for hermetic tests that
    must not leak a custom vocabulary into the rest of the suite.
    """

    def __init__(self, lexicon: Lexicon):
        self._new = lexicon
        self._prev: Optional[Lexicon] = None

    def __enter__(self) -> Lexicon:
        global _active
        self._prev = _active
        _active = self._new
        return self._new

    def __exit__(self, *exc) -> None:
        global _active
        _active = self._prev


__all__ = [
    "Lexicon",
    "get_active_lexicon",
    "set_active_lexicon",
    "use_lexicon",
]
