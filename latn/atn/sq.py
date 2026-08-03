"""SQ -- the interrogative clause ATN.

Recognises a wh-question ("what color is the arch", "where is the tavern") or a
subject-inverted yes/no question ("is the box on the table", "are your bolts for
sale") and captures the wh lexeme / yes-no flag, the queried focus NP, an
optional predicate VP, and any prepositional relation.

Its START state accepts ONLY a wh-word or a leading to-be verb, so it produces
no parse for declaratives or imperatives. That is deliberate: SQ is meant to run
as a *parallel* layer-5 pass beside the SP ATN, which stays untouched -- so a
question surfaces its own SQ parse while every other sentence is unaffected.
"""

from latn.lexer.token_stream import TokenStream
from latn.atn.core import ATNState, noop
from latn.utils.predicates import (
    is_wh, is_tobe, is_verb, is_np_token, is_vp_token, is_pp_token,
    is_adjective, is_none, is_anything_no_consume, is_expletive,
)
from latn.pos.question_phrase import QuestionPhrase


def build_sq_atn(q: QuestionPhrase, ts: TokenStream):
    start = ATNState("SQ-START")
    body = ATNState("SQ-BODY")
    end = ATNState("SQ-END")

    def is_inverted_auxiliary(tok):
        if tok is None or not is_vp_token(tok) or not tok.isa("aux"):
            return False
        vp = getattr(tok, "phrase", None)
        subject = getattr(vp, "noun_phrase", None)
        return bool(subject is not None and (
            getattr(subject, "noun", None)
            or getattr(subject, "pronoun", None)
            or getattr(subject, "proper_noun", None)
        ))

    # A wh-word, inverted to-be, or a modal/auxiliary VP with an embedded
    # subject opens an interrogative.
    start.add_arc(is_wh, lambda _, tok: q.apply_wh(tok), body)
    start.add_arc(is_inverted_auxiliary,
                  lambda _, tok: q.mark_inverted_auxiliary(tok), body)
    start.add_arc(is_tobe, lambda _, tok: q.mark_yesno(tok), body)

    # Capture the first NP as the queried focus, fold in a predicate VP (which
    # itself may carry the focus as its object) / PP / adjective, and absorb any
    # bare auxiliary/verb. A copular VP token also satisfies is_tobe/is_verb, so
    # is_vp_token MUST be tried before them or the VP -- and the focus inside it
    # -- gets eaten as a bare to-be. Every looping arc CONSUMES (never noop) so
    # the state can't spin; anything unrecognised ends the question and is left
    # for the outer parse.
    body.add_arc(is_np_token, lambda _, tok: q.apply_focus_token(tok), body)
    body.add_arc(is_vp_token, lambda _, tok: q.apply_predicate_token(tok), body)
    body.add_arc(is_pp_token, lambda _, tok: q.apply_prepositional_phrase(tok), body)
    body.add_arc(is_tobe, lambda _, tok: None, body)
    body.add_arc(is_verb, lambda _, tok: None, body)
    body.add_arc(is_adjective, lambda _, tok: None, body)
    body.add_arc(is_expletive, lambda _, tok: None, body)
    body.add_arc(is_none, noop, end)
    body.add_arc(is_anything_no_consume, noop, end)

    end.add_arc(is_none, noop, end)
    return start, end
