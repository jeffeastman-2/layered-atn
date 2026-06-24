# engraf/atn/subnet_pp.py
from latn.atn.pp import build_pp_atn
from latn.lexer.token_stream import TokenStream
from latn.atn.core import run_atn
from latn.pos.prepositional_phrase import PrepositionalPhrase

def run_pp(tokens):
    ts = TokenStream(tokens)
    pp = PrepositionalPhrase()
    pp_start, pp_end = build_pp_atn(pp,ts)
    result = run_atn(pp_start, pp_end, ts, pp)
    return result
