from latn.lexer.token_stream import TokenStream
from latn.atn.core import ATNState, noop
from latn.utils.predicates import is_anything_no_consume, is_pp_token, is_quoted, is_conjunction, \
    is_none, is_adverb, is_adjective, is_np_token, is_vp_token
from latn.pos.sentence_phrase import SentencePhrase

def build_sentence_atn(sent: SentencePhrase, ts: TokenStream):
    start = ATNState("SENTENCE-START")
    predicate = ATNState("SENTENCE-PREDICATE")
    adjective_phrase = ATNState("SENTENCE-ADJECTIVE-PHRASE")
    adjective = ATNState("SENTENCE-ADJECTIVE")
    adverb = ATNState("SENTENCE-ADVERB")
    conj = ATNState("SENTENCE-CONJ")  # e.g., "and transparent"
    end = ATNState("SENTENCE-END")

    # Optional subject NP at start
    start.add_arc(is_np_token, lambda _, tok: sent.apply_subject_token(tok), predicate)
    start.add_arc(is_quoted, lambda _, tok: sent.store_definition_word(tok), predicate) 
    start.add_arc(is_anything_no_consume, noop,  predicate) 

    predicate.add_arc(is_vp_token, lambda _, tok: sent.apply_predicate_token(tok), end)
    predicate.add_arc(is_none, noop, end)  # Allow empty predicate

    # Start by accepting an adverb (optional) or adjective
    adjective_phrase.add_arc(is_pp_token, lambda _, tok: sent.apply_prepositional_phrase(tok), end)
    adjective_phrase.add_arc(is_adverb, lambda _, tok: sent.apply_adverb(tok), adverb)
    adjective_phrase.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # Allow chain of adverbs before adjectives
    adverb.add_arc(is_adverb, lambda _, tok: sent.apply_adverb(tok), adverb)
    adverb.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # After adjective, allow conjunction for chained descriptions (e.g. "and rough")
    adjective.add_arc(is_conjunction, lambda _, tok: None, conj)  # Consume the conjunction
    conj.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # End of adjective phrase
    adjective.add_arc(is_none, noop, end)

    # Allow final transition if stream is exhausted
    end.add_arc(is_none, noop, end)

    return start, end
