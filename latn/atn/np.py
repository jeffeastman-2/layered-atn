import numpy as np
from latn.lexer.token_stream import TokenStream
from latn.utils.predicates import any_of, is_verb, is_adverb, is_noun, is_tobe, \
    is_determiner, is_pronoun, is_adjective, is_preposition, is_none, is_vector, \
        is_conjunction, is_number, is_unknown, is_negation, is_punctuation, \
        is_proper_noun
from latn.atn.core import ATNState, noop
from latn.pos.noun_phrase import NounPhrase

# --- Build the Noun Phrase ATN ---
def build_np_atn(np: NounPhrase, ts: TokenStream):
    # Helper function to check if token is a non-comparative adjective
    def is_non_comparative_adjective(tok):
        return is_adjective(tok) and tok.scalar_projection("comp") == 0.0

    def is_adjective_modifier(tok):
        """True when an adjective-typed token should be consumed as a MODIFIER
        rather than the noun head.

        A pure adjective is always a modifier. A word typed as BOTH adj and
        noun (common in domain lexicons — "mace", "dwarf", "rocky chamber") is
        a modifier only when another noun-capable word follows it; if a
        non-noun follows (a preposition, end of stream, a verb, ...), the dual
        word is itself the NP's noun head, so this arc declines and the is_noun
        arc takes it. Lookahead over the token stream picks the head as the
        last dual word in a run. The NP vector accumulates the same dims
        whether the word is applied as adj or noun, so nothing is lost.
        """
        if not is_adjective(tok):
            return False
        if not is_noun(tok):
            return True  # pure adjective: always a modifier
        nxt = ts.tokens[ts.position + 1] if ts.position + 1 < len(ts.tokens) else None
        return bool(nxt and (is_noun(nxt) or is_proper_noun(nxt)))

    def is_noun_modifier(tok):
        """A noun acting as a pre-head modifier in a noun-noun compound: a noun
        immediately followed by another noun-capable word, so it modifies a
        later head ("passage mouth", "low shadow", "echo chamber arch"). The
        last noun in the run is the head -- this declines for it so the is_noun
        arc takes it. apply_noun and apply_adjective accumulate the same vector
        dims, so consuming a modifier noun as a noun loses nothing. Domain
        lexicons have many noun-noun compounds; Engraf's CAD vocab has none, so
        this is inert for Engraf."""
        if not is_noun(tok):
            return False
        nxt = ts.tokens[ts.position + 1] if ts.position + 1 < len(ts.tokens) else None
        return bool(nxt and (is_noun(nxt) or is_proper_noun(nxt)))

    start = ATNState("NP-START")
    det = ATNState("NP-DET")
    adj = ATNState("NP-ADJ")
    adj_conj = ATNState("NP-ADJ-CONJ")  # New state for handling conjunctions between adjectives
    adj_after_pronoun = ATNState("NP-ADJ-AFTER-PRONOUN")
    noun = ATNState("NP-NOUN")
    end = ATNState("NP-END")

    start.add_arc(is_determiner, lambda _, tok: np.apply_determiner(tok), det)
    start.add_arc(is_pronoun, lambda _, tok: np.apply_pronoun(tok), adj_after_pronoun)
    start.add_arc(is_vector, lambda _, tok: np.apply_vector(tok), end)
    
    # LATN Extension: Allow NPs to start with adjectives, adverbs, or nouns
    start.add_arc(is_adjective_modifier, lambda _, tok: np.apply_adjective(tok), adj)
    start.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), det)  # "very" -> DET state to handle "very big sphere"
    # A noun that modifies a following noun (compound) routes to ADJ to keep
    # looking for the head; a lone/final noun falls through to the head arc.
    start.add_arc(is_noun_modifier, lambda _, tok: np.apply_noun(tok), adj)
    start.add_arc(is_noun, lambda _, tok: np.apply_noun(tok), noun)     # Allow bare nouns like "sphere"
    # Allow a proper noun as an NP head ("Amara", "Sir Roderick"). Reuses
    # apply_noun (sets noun/vector/consumed_tokens); the token's proper_noun
    # POS rides into the NP vector. Distinct from apply_proper_noun, which is
    # the naming-syntax directive ("call it 'Charlie'").
    start.add_arc(is_proper_noun, lambda _, tok: np.apply_noun(tok), noun)

    # ADJ → ADJ / NOUN
    det.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), det)
    det.add_arc(is_adjective_modifier, lambda _, tok: np.apply_adjective(tok), adj)
    # Terminate on unknown tokens after determiners
    det.add_arc(is_unknown, noop, end)

    adj.add_arc(is_adjective_modifier, lambda _, tok: np.apply_adjective(tok), adj)
    # Handle adverbs that modify subsequent adjectives (e.g., "small bright blue")
    adj.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj)
    # Handle conjunctions between adjectives (e.g., "tall and red") - but NOT noun phrase coordination
    def is_adjective_conjunction(tok):
        if not is_conjunction(tok):
            return False
        # Peek ahead to see if next token is an adjective (not a determiner starting new NP)
        current_pos = ts.position
        next_tok = None
        if current_pos + 1 < len(ts.tokens):
            next_tok = ts.tokens[current_pos + 1]
        return next_tok and is_adjective(next_tok) and not is_determiner(next_tok)
    
    adj.add_arc(is_adjective_conjunction, lambda _, tok: None, adj_conj)  # Consume the conjunction
    
    # After conjunction, expect another adjective (possibly with adverb modifier)
    adj_conj.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_conj)
    adj_conj.add_arc(is_adjective_modifier, lambda _, tok: np.apply_adjective(tok), adj)

    adj_after_pronoun.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_after_pronoun)
    # Allow adjectives after pronouns, but NOT comparative adjectives (let VP handle those)
    adj_after_pronoun.add_arc(is_non_comparative_adjective, lambda _, tok: np.apply_adjective(tok), adj_after_pronoun)
    # Handle conjunctions between adjectives after pronouns - but only if followed by adjective
    def is_conjunction_followed_by_adjective(tok):
        if not is_conjunction(tok):
            return False
        # Peek ahead to see if next token is an adjective
        current_pos = ts.position
        next_tok = None
        if current_pos + 1 < len(ts.tokens):
            next_tok = ts.tokens[current_pos + 1]
        return next_tok and is_adjective(next_tok)
    
    adj_after_pronoun.add_arc(is_conjunction_followed_by_adjective, lambda _, tok: None, adj_conj)
    # End on various boundary conditions but don't consume tokens
    adj_after_pronoun.add_arc(any_of(is_verb, is_tobe, is_conjunction, is_adjective), noop, end)
    adj_after_pronoun.add_arc(is_none, noop, end)
    # Terminate on unknown tokens after pronouns - this ensures pronouns can end properly
    adj_after_pronoun.add_arc(is_unknown, noop, end)
    # Also end on prepositions and numbers that might follow pronouns
    adj_after_pronoun.add_arc(any_of(is_preposition, is_number), noop, end)

    # ADJ or DET → NOUN (proper noun also accepted as head: "the dwarf Amara").
    # A noun that modifies a following noun (compound) routes back to ADJ to
    # keep looking for the head; the final noun falls through to the head arc.
    for state in [det, adj, adj_conj]:
        state.add_arc(is_noun_modifier, lambda _, tok: np.apply_noun(tok), adj)
        state.add_arc(is_noun, lambda _, tok: np.apply_noun(tok), noun)
        state.add_arc(is_proper_noun, lambda _, tok: np.apply_noun(tok), noun)

    # Allow ADJ state to end on various conditions
    adj.add_arc(is_none, noop, end)
    adj.add_arc(any_of(is_verb, is_tobe, is_conjunction), noop, end)
    # Terminate on unknown tokens after adjectives
    adj.add_arc(is_unknown, noop, end)

    # NOUN → END (simple NP)
    noun.add_arc(is_none, noop, end)
    # NOUN → END (when conjunction is encountered - don't consume the conjunction)
    noun.add_arc(is_conjunction, noop, end)
    # NOUN → END (when verb, tobe, adjective, or preposition is encountered - don't consume)
    noun.add_arc(any_of(is_verb, is_tobe, is_adjective, is_preposition), noop, end)
    # NOUN → END (when determiner is encountered - don't consume, starts new NP)
    noun.add_arc(is_determiner, noop, end)
    # NOUN → END (when pronoun is encountered - don't consume, starts new NP)
    noun.add_arc(is_pronoun, noop, end)
    # NOUN → END (when number is encountered - don't consume, starts new NP)
    noun.add_arc(is_number, noop, end)
    # NOUN → END (when adverb is encountered - don't consume, likely modifies verb)
    noun.add_arc(is_adverb, noop, end)
    # NOUN → END (when vector is encountered - don't consume, likely a separate NP)
    noun.add_arc(is_vector, noop, end)
    # NOUN → END (when negation token is encountered - don't consume, likely a separate NP)
    noun.add_arc(is_negation, noop, end)
    # NOUN → END (when punctuation is encountered - don't consume)
    noun.add_arc(is_punctuation, noop, end)
    # Terminate on unknown tokens after nouns - this is the key fix
    noun.add_arc(is_unknown, noop, end)

    return start, end


def run_np(tokens):
    """Run the NP ATN on a sequence of tokens.
    
    Args:
        tokens: List of VectorSpace tokens
        
    Returns:
        NounPhrase object if successful, None if parsing fails
    """
    from latn.atn.core import run_atn
    
    ts = TokenStream(tokens)
    np = NounPhrase()
    np_start, np_end = build_np_atn(np, ts)
    result = run_atn(np_start, np_end, ts, np)
    return result


