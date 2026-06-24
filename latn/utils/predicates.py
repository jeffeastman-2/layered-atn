def is_determiner(tok): return tok is not None and tok.isa("det")
def is_pronoun(tok): return tok is not None and tok.isa("pronoun")
def is_adjective(tok): return tok is not None and tok.isa("adj")
def is_adverb(tok): return tok is not None and tok.isa("adv")
def is_noun(tok): return tok is not None and tok.isa("noun")
def is_proper_noun(tok): return tok is not None and tok.isa("proper_noun")
def is_verb(tok): return tok is not None and tok.isa("verb")
def is_tobe(tok): return tok is not None and tok.isa("tobe")
def is_preposition(tok): return tok is not None and tok.isa("prep")
def is_negation(tok): return tok is not None and tok.isa("neg")
def is_punctuation(tok): return tok is not None and tok.isa("punct")
def is_number(tok): return tok is not None and tok["number"] > 0.0
def is_none(tok): return tok is None
def is_vector(tok): return tok is not None and tok.isa("vector")
def is_quoted(tok): return tok is not None and tok.isa("quoted")
def is_adjective_or_adverb(tok): return tok is not None and (tok.isa("adj") or tok.isa("adv"))
def is_np_token(tok): return tok is not None and tok.isa("NP")
def is_pp_token(tok): return tok is not None and tok.isa("PP")
def is_vp_token(tok): 
    return tok is not None and tok.isa("VP")
def is_unknown(tok): return tok is not None and tok.isa("unknown")
def any_of(*predicates):
    def combined_predicate(tok):
        return any(pred(tok) for pred in predicates)
    return combined_predicate
def is_any_of(tok, predicates):
    for pred in predicates:
        if tok.isa(pred):
            return True
    return False
def is_conjunction(tok): return tok is not None and tok.isa("conj")
def is_anything(tok): return True
def is_anything_no_consume(tok):
    return True, False
def is_conjunction_no_consume(tok):
    return is_conjunction(tok), False
def is_noun_phrase_token(tok):
    """Returns True if the token is a NounPhrase token from Layer 2."""
    return tok is not None and tok.isa("NP")

def is_np_head(tok):
    """Returns True if the token starts a noun phrase."""
    return any_of(is_determiner, is_pronoun, is_vector, is_noun_phrase_token)(tok)

def is_adjective_conjunction(ts):
    """Returns a predicate that checks if current token is 'and' followed by adjective."""
    def predicate(tok):
        if not is_conjunction(tok):
            return False
        
        # Look ahead to see if next token is an adjective
        current_pos = ts.position
        next_tok = None
        if current_pos + 1 < len(ts.tokens):
            next_tok = ts.tokens[current_pos + 1]
        
        return next_tok is not None and is_adjective(next_tok)
    
    return predicate

def is_predicate_conjunction(ts):
    """Returns a predicate that checks if current token is 'and' NOT followed by adjective."""
    def predicate(tok):
        if not is_conjunction(tok):
            return False, False
        
        # Look ahead to see if next token is an adjective
        current_pos = ts.position
        next_tok = None
        if current_pos + 1 < len(ts.tokens):
            next_tok = ts.tokens[current_pos + 1]
        
        # If next token is not an adjective, this is likely a predicate conjunction
        is_predicate_conj = next_tok is None or not is_adjective(next_tok)
        return is_predicate_conj, False  # Don't consume the conjunction
    
    return predicate

def is_conjunction_only(tok):
    return tok is not None and tok.isa("conj") and not (tok.isa("NP") or tok.isa("PP") or tok.isa("VP"))