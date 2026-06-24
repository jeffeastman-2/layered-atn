from latn.lexer.token_stream import TokenStream
from latn.utils.predicates import is_preposition, is_noun_phrase_token,is_negation    
from latn.atn.core import ATNState
from latn.pos.prepositional_phrase import PrepositionalPhrase

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(pp:PrepositionalPhrase, ts:TokenStream):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-AFTER-PREP")
    end = ATNState("PP-END")

    # PREP
    start.add_arc(is_preposition, lambda _, tok: pp.apply_preposition(tok), after_prep)
    start.add_arc(is_negation, lambda _, tok: pp.apply_negation(tok), start)
    
    # Handle NounPhrase tokens from Layer 2 - preserve grounded scene objects
    def apply_grounded_np(pp_obj, tok):
        # Use grounded phrase if available (from Layer 2 semantic grounding), otherwise original NP
        try:
            grounded_phrase = tok.phrase
            if grounded_phrase is not None:
                pp_obj.apply_np(grounded_phrase)
                return
        except AttributeError:
            pass
        
        try:
            original_np = tok._original_np
            pp_obj.apply_np(original_np)
        except AttributeError:
            pass
    
    after_prep.add_arc(is_noun_phrase_token, apply_grounded_np, end)

    return start, end
