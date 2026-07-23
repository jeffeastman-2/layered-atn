from latn.atn.core import ATNState
from latn.lexer.token_stream import TokenStream
from latn.utils.predicates import is_adverb, is_verb, is_none, is_np_token, \
    is_pp_token, is_conjunction_no_consume, any_of, is_tobe, is_adjective, \
        is_conjunction_only, is_anything_no_consume, is_negation
from latn.pos.verb_phrase import VerbPhrase
from latn.atn.core import ATNState, noop


# --- Build the Verb Phrase ATN ---
def build_vp_atn(vp: VerbPhrase, ts: TokenStream):
    start = ATNState("VP-START")
    after_verb = ATNState("VP-AFTER-VERB")
    after_tobe = ATNState("VP-AFTER-TOBE")
    after_adverb = ATNState("VP-AFTER-ADVERB")
    after_np = ATNState("VP-AFTER-NP")
    after_adj = ATNState("VP-AFTER-ADJ")
    after_pp = ATNState("VP-AFTER-PP")
    end = ATNState("VP-END")

    # VERB
    start.add_arc(is_adverb, lambda _, tok: vp.apply_adverb(tok), after_adverb)
    start.add_arc(is_verb, lambda _, tok: vp.apply_verb(tok), after_verb)
    start.add_arc(is_tobe, lambda _, tok: vp.apply_verb(tok), after_tobe)

    after_adverb.add_arc(is_verb, lambda _, tok: vp.apply_verb(tok), after_verb)
    after_adverb.add_arc(is_adverb, lambda _, tok: vp.apply_adverb(tok), after_adverb)

    # Look for a conjunction as in "slightly rotate and move" then go back for another verb
    after_verb.add_arc(is_conjunction_only, lambda _, tok: vp.apply_conjunction(tok), start)
    after_verb.add_arc(is_anything_no_consume, noop, after_tobe)  # Allow VP to end after verb if nothing follows

    # After VERB: look for NP tokens (created by Layer 2)
    after_tobe.add_arc(is_negation, lambda _, tok: vp.apply_negation(tok), after_tobe)
    after_tobe.add_arc(is_np_token, lambda _, tok: vp.apply_np(_extract_np_from_token(tok)), after_np)
    # ... or PP tokens created by Layer 3 (e.g., "is above the cube")
    after_tobe.add_arc(is_pp_token, lambda _, tok: vp.apply_pp(_extract_pp_from_token(tok)), after_pp)      
    # For "to be" verbs, allow direct adjectives
    after_tobe.add_arc(is_adjective, lambda _, tok: vp.apply_adjective(tok), after_adj)
    # Allow final transition if stream is exhausted
    after_tobe.add_arc(is_none, noop, end)

    # After NP: can have PP tokens (created by Layer 3), adjective complement, or end
    after_np.add_arc(is_pp_token, lambda _, tok: vp.apply_pp(_extract_pp_from_token(tok)), after_pp)
    after_np.add_arc(is_adjective, lambda _, tok: vp.apply_adjective(tok), after_adj)
    after_np.add_arc(is_none, noop, end)
    # Allow VP to end when conjunction is encountered (don't consume it)
    after_np.add_arc(is_conjunction_no_consume, noop, end)
    # Allow VP to end when other verbs/tobe are encountered
    after_np.add_arc(any_of(is_verb, is_tobe), noop, end)
    # A completed verb+object VP is done: any *other* trailing token -- an unknown
    # word, a stray adverb ("now"), punctuation -- reduces to end and is left for
    # the outer parse, instead of dead-ending the whole VP. Last arc, so the
    # specific continuations above are always tried first.
    after_np.add_arc(is_anything_no_consume, noop, end)

    # After PP: can have more PPs, adjectives, or end
    after_pp.add_arc(is_pp_token, lambda _, tok: vp.apply_pp(_extract_pp_from_token(tok)), after_pp)
    after_pp.add_arc(is_adjective, lambda _, tok: vp.apply_adjective(tok), after_adj)
    after_pp.add_arc(is_none, noop, end)
    after_pp.add_arc(is_conjunction_no_consume, noop, end)
    after_pp.add_arc(any_of(is_verb, is_tobe), noop, end)
    after_pp.add_arc(is_anything_no_consume, noop, end)   # stray trailing token -> reduce (see after_np)

    # After adjective complement: can have more adjectives or end
    after_adj.add_arc(is_adjective, lambda _, tok: vp.apply_adjective(tok), after_adj)
    after_adj.add_arc(is_none, noop, end)
    after_adj.add_arc(is_conjunction_no_consume, noop, end)
    after_adj.add_arc(any_of(is_verb, is_tobe), noop, end)
    after_adj.add_arc(is_anything_no_consume, noop, end)   # stray trailing token -> reduce (see after_np)

    return start, end


def _extract_np_from_token(np_token):
    """Extract or create a NounPhrase object from an NP token."""
    from latn.pos.noun_phrase import NounPhrase

    if np_token.phrase is not None:
        return np_token.phrase

    # Create a simple NounPhrase from the token
    np = NounPhrase()
    if hasattr(np_token, 'word') and np_token.word.startswith("NP("):
        np_text = np_token.word[3:-1]  # Remove "NP(" and ")"
        words = np_text.split()
        if words:
            np.head_noun = words[-1]  # Last word is usually the noun
    np.vector = np_token
    return np


def _extract_pp_from_token(pp_token):
    """Extract or create a PrepositionalPhrase object from a PP token."""
    from latn.pos.prepositional_phrase import PrepositionalPhrase
    
    if pp_token.phrase is not None:
        return pp_token.phrase

    # Create a simple PrepositionalPhrase from the token
    pp = PrepositionalPhrase()
    if hasattr(pp_token, 'word') and pp_token.word.startswith("PP("):
        pp_text = pp_token.word[3:-1]  # Remove "PP(" and ")"
        words = pp_text.split()
        if words:
            pp.preposition = words[0]  # First word is usually the preposition
    pp.vector = pp_token
    return pp


