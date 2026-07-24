"""SQ -- the interrogative clause ATN, exercised host-neutrally.

The SQ ATN is a standalone parallel layer-5 pass: given the layer-4 token stream
it recognises a wh- or yes/no question and captures its structure (wh lexeme /
yes-no flag, the queried focus NP, prepositional relations), while producing NO
parse for a declarative or an imperative. These tests prove both halves on the
neutral test vocabulary -- no host semantics involved.
"""

import pytest

from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.lexer.token_stream import TokenStream
from latn.atn.core import run_atn
from latn.atn.sq import build_sq_atn
from latn.pos.question_phrase import QuestionPhrase


pytestmark = pytest.mark.usefixtures("neutral_latn")


def _parse_sq(sentence):
    """Run layer-4, then the SQ ATN from the start of the token stream.

    Returns the QuestionPhrase when the utterance parses as a question, else
    None -- mirroring how the parallel layer-5 SQ pass will be wired.
    """
    result = LATNLayerExecutor().execute_layer4(sentence)
    tokens = list(result.hypotheses[0].tokens) if result.hypotheses else []
    ts = TokenStream(tokens)
    q = QuestionPhrase()
    accepted = run_atn(*build_sq_atn(q, ts), ts, q)
    return q if accepted is not None else None


def _focus_noun(q):
    return (getattr(q.focus, "noun", None) or "").lower() if q and q.focus else None


# --- wh questions -----------------------------------------------------------

def test_wh_copular_captures_wh_and_focus():
    q = _parse_sq("what is it")
    assert q is not None
    assert q.wh_word == "what"
    assert not q.is_yesno
    # "is it" folds into a copular VP; the focus is that object pronoun NP.
    assert q.focus is not None
    assert "it" in q.focus.printString().lower()


def test_who_question():
    q = _parse_sq("who is it")
    assert q is not None
    assert q.wh_word == "who"
    assert q.focus is not None


# --- yes/no questions -------------------------------------------------------

def test_yesno_copular_captures_focus_noun():
    q = _parse_sq("is the box on the table")
    assert q is not None
    assert q.is_yesno
    assert q.wh_word is None
    assert _focus_noun(q) == "box"


def test_yesno_focus_and_relation():
    q = _parse_sq("is the box on the table")
    # The prepositional relation rides along for the host to resolve.
    assert q.prepositional_phrases, "expected the 'on the table' relation captured"


# --- the zero-regression property: non-questions must NOT parse as SQ --------

def test_imperative_is_not_a_question():
    assert _parse_sq("inspect the box") is None


def test_declarative_is_not_a_question():
    assert _parse_sq("the box is red") is None
