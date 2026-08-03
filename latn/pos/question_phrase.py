"""SQ -- the interrogative sentence nonterminal.

A distinct clause type from SP (which is declarative/imperative). An SQ is either
a wh-question ("what color is the arch", "where is the tavern") or a subject-
inverted yes/no question ("is the box on the table", "are your bolts for sale").

It carries the *structure* of the query -- the wh lexeme (or a yes/no flag), the
queried focus NP, an optional predicate VP, and any prepositional relation --
and nothing about what the answer is: grounding and answering belong to the host
(the same Achem's-Razor split the verb path uses). Built by ``build_sq_atn``.
"""

from latn.lexer.vector_space import VectorSpace


def _is_referential(np) -> bool:
    """True if an NP can serve as a question's subject/focus -- it has a noun
    head or is a pronoun/proper noun. A copula's bare adjective complement
    ("are [red]") is wrapped as a noun_phrase but is NOT referential, so it must
    not be mistaken for the subject."""
    if np is None:
        return False
    return bool(getattr(np, "noun", None)
                or getattr(np, "pronoun", None)
                or getattr(np, "proper_noun", None))


class QuestionPhrase:
    def __init__(self):
        self.vector = VectorSpace()
        self.wh_word = None          # "what"/"where"/... or None for a yes/no
        self.is_yesno = False        # True when led by an inverted to-be/aux
        self.focus = None            # the queried NP (subject of the question)
        self.subject = None          # grammatical subject under aux/modal inversion
        self.auxiliary = None        # leading inverted auxiliary/modal VP
        self.attribute = None        # queried property NP ("what COLOR is the arch")
        self.predicate = None        # optional predicate VP
        self.prepositional_phrases = []   # e.g. "on the table"
        self._focus_is_bare = False  # focus came from a bare NP, not a VP object

    # --- ATN actions ---
    def apply_wh(self, tok):
        """A leading wh-word: record which wh it is."""
        word = getattr(tok, "word", None)
        self.wh_word = word.lower() if isinstance(word, str) else word
        self.vector += tok

    def mark_yesno(self, tok):
        """A leading inverted auxiliary (to-be/do): a yes/no question. When the
        lead has already absorbed its subject into a copular VP -- "are [your
        bolts]" -- lift that object out as the queried focus."""
        self.is_yesno = True
        self.vector += tok
        self._absorb_vp(tok)

    def mark_inverted_auxiliary(self, tok):
        """Capture ``can I`` / ``do they`` without mistaking the subject for
        the queried object. The following predicate VP supplies the focus."""
        self.is_yesno = True
        self.vector += tok
        vp = getattr(tok, "phrase", None)
        self.auxiliary = vp
        subject = getattr(vp, "noun_phrase", None) if vp is not None else None
        if _is_referential(subject):
            self.subject = subject

    def apply_focus_token(self, token):
        """A bare NP after the lead is the queried focus/subject (provisionally;
        a following copula's object can promote this to the ATTRIBUTE)."""
        if self.focus is None:
            self.focus = getattr(token, "phrase", None) or token
            self._focus_is_bare = True

    def apply_predicate_token(self, token):
        """A predicate VP. A copular/auxiliary VP folds the queried subject
        inside itself as its object NP -- "is [it]", "are [your bolts]" -- so
        lift that object out as the focus, and keep the VP as the predicate."""
        self._absorb_vp(token)

    def _absorb_vp(self, token):
        """Pull the focus (the VP's object NP) and predicate out of a VP token."""
        vp = getattr(token, "phrase", None)
        if vp is None:
            return
        obj = getattr(vp, "noun_phrase", None)
        if _is_referential(obj):
            if self.focus is None:
                self.focus = obj
            elif self._focus_is_bare and self.attribute is None:
                # "what color is the arch": the bare NP after the wh ("color")
                # was the queried ATTRIBUTE; the copula's object ("the arch") is
                # the subject the host reads that attribute off of.
                self.attribute = self.focus
                self.focus = obj
                self._focus_is_bare = False
        if self.predicate is None:
            self.predicate = vp
        # A copular VP swallows trailing PPs into itself ("is the box [on the
        # table]"); surface them as the question's relations for the host.
        for pp in getattr(vp, "prepositions", None) or ():
            self.prepositional_phrases.append(pp)

    def apply_prepositional_phrase(self, pp_token):
        self.prepositional_phrases.append(pp_token)
        return True

    # --- nonterminal interface (mirrors SentencePhrase, used by the tokenizer) ---
    def to_vector(self):
        return self.vector

    def descriptive_word(self):
        kind = self.wh_word or ("yesno" if self.is_yesno else "?")
        return f"SQ({kind})"

    @staticmethod
    def phrase_type():
        return "SQ"

    def __repr__(self):
        return (f"Question(wh={self.wh_word!r}, yesno={self.is_yesno}, "
                f"subject={self.subject}, "
                f"attribute={self.attribute}, focus={self.focus}, "
                f"preps={len(self.prepositional_phrases)})")
