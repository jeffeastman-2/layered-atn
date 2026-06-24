# engraf/atn/subnet_pp.py
from latn.atn.vp import build_vp_atn
from latn.lexer.token_stream import TokenStream
from latn.atn.core import run_atn
from latn.pos.verb_phrase import VerbPhrase

def run_vp(tokens):
    ts = TokenStream(tokens)
    vp = VerbPhrase()
    vp_start, vp_end = build_vp_atn(vp, ts)
    return run_atn(vp_start, vp_end, ts, vp)
