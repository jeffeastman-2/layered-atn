from latn.atn.core import ATNState, run_atn, noop
from latn.atn.sentence import build_sentence_atn
from latn.lexer.token_stream import TokenStream
from latn.pos.sentence_phrase import SentencePhrase



def run_sentence(tokens):
    ts = TokenStream(tokens)
    sent = SentencePhrase()
    s_start, s_end = build_sentence_atn(sent, ts)
    return run_atn(s_start, s_end, ts, sent)
