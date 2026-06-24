# engraf/atn/subnet_np.py
from latn.atn.np import build_np_atn
from latn.lexer.token_stream import TokenStream
from latn.atn.core import run_atn
from latn.pos.noun_phrase import NounPhrase


def run_np(tokens):
    ts = TokenStream(tokens)
    np = NounPhrase()
    np_start, np_end = build_np_atn(np, ts)
    return run_atn(np_start, np_end, ts, np)
