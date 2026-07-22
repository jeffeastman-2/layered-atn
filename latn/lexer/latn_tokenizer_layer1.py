"""
LATN (Layered Augmented Transition Network) Lexical Layer

This module implements multi-hypothesis tokenization that returns ranked alternatives
instead of committing early to a single tokenization.
"""

from latn.lexer.lexicon import get_active_lexicon
from latn.lexer.vocabulary_builder import vector_from_word
from latn.lexer.vector_space import VectorSpace, vector_from_features
from latn.lexer.hypothesis import TokenizationHypothesis
from latn.utils.noun_inflector import singularize_noun
from latn.utils.verb_inflector import find_root_verb
import re
from typing import List, Tuple


def latn_tokenize_layer1(sentence: str) -> List[TokenizationHypothesis]:
    """
    LATN lexical layer: Return multiple ranked tokenization hypotheses.
    
    Args:
        sentence: Input sentence string
        
    Returns:
        List of TokenizationHypothesis objects, ranked by confidence (highest first)
    """
    # First, extract raw tokens using the same regex as original
    pattern = re.compile(
        r"""\s*(
            \[             # opening bracket
            \s*-?\d+(\.\d+)?\s*,     # x
            \s*-?\d+(\.\d+)?\s*,     # y
            \s*-?\d+(\.\d+)?\s*      # z
            \]             # closing bracket
            | -?\d+(?:\.\d+)?         # standalone number (int or float)
            | '[\w\s]+'     # quoted words (single quotes)
            | \w+(?:-\w+)*  # normal word (including hyphenated words)
            | [^\w\s]       # punctuation
        )""",
        re.VERBOSE,
    )
    tokens = pattern.findall(sentence)
    flat_tokens = [t[0] for t in tokens]
    
    # Generate all possible tokenization hypotheses
    hypotheses = []
    
    # Generate hypotheses recursively
    def generate_hypotheses(token_index: int, current_tokens: List[VectorSpace], current_confidence: float, description: str):
        """Recursively generate all possible tokenization hypotheses."""
        
        if token_index >= len(flat_tokens):
            # Reached end - add this hypothesis
            hypotheses.append(TokenizationHypothesis(
                tokens=current_tokens.copy(), 
                confidence=current_confidence,
                description=description
            ))
            return
        
        # Try different grouping options at current position
        remaining_tokens = len(flat_tokens) - token_index
        max_group_size = min(3, remaining_tokens)  # Try up to 3-word compounds
        
        for group_size in range(1, max_group_size + 1):
            if token_index + group_size > len(flat_tokens):
                continue
                
            # Extract the token group
            if group_size == 1:
                tok = flat_tokens[token_index]
                compound_key = tok.lower()
            else:
                tok_group = flat_tokens[token_index:token_index + group_size]
                compound_key = " ".join(tok_group).lower()
                tok = compound_key
            
            # Try to process this token/compound
            processed_token, confidence_boost, process_description = process_token_group(tok, group_size)
            
            if processed_token is not None:
                # Calculate confidence for this choice
                # Use average confidence rather than sum to avoid bias toward more tokens
                decision_confidence = current_confidence * len(current_tokens) + confidence_boost
                new_confidence = decision_confidence / (len(current_tokens) + 1)
                new_description = f"{description} | {process_description}" if description else process_description
                
                # Recursively generate the rest
                generate_hypotheses(
                    token_index + group_size,
                    current_tokens + [processed_token],
                    new_confidence,
                    new_description
                )
    
    # Start recursive generation
    generate_hypotheses(0, [], 0.0, "")
    
    # Sort by confidence (descending) and return
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return hypotheses


def process_token_group(tok: str, group_size: int) -> Tuple[VectorSpace, float, str]:
    """
    Process a token or token group, returning the VectorSpace, confidence boost, and description.
    
    Args:
        tok: Token string (single word or multi-word compound)
        group_size: Number of original tokens this represents
        
    Returns:
        (VectorSpace object, confidence_boost, description) or (None, 0, "") if invalid
    """
    
    # Handle special tokens first (vectors, quoted strings, numbers)
    if tok.startswith("'") and tok.endswith("'"):
        inner_tok = tok[1:-1]
        vs = vector_from_features(pos="quoted")
        vs.word = inner_tok
        return vs, 0.8, f"quoted-string({inner_tok})"
    
    elif tok.startswith("[") and tok.endswith("]"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", tok)
        if len(nums) == 3:
            x, y, z = map(float, nums)
            from latn.lexer.literal_decoder import get_active_literal_decoder
            vs = get_active_literal_decoder().decode((x, y, z))
            vs.word = tok
            return vs, 1.0, f"vector({x},{y},{z})"
        return None, 0, ""
    
    elif re.fullmatch(r"-?\d+(?:\.\d+)?", tok):
        vs = vector_from_features(pos="det def number")
        vs.word = tok
        vs["number"] = float(tok)
        return vs, 0.9, f"number({tok})"
    
    # Handle word tokens
    else:
        # Check for morphological inflections BEFORE vocabulary lookup
        singular_form, was_plural = singularize_noun(tok)
        lookup_word = singular_form.lower()
        
        # Try vocabulary lookup with singular form
        if lookup_word in get_active_lexicon():
            vs = vector_from_word(lookup_word)
            vs.word = singular_form
            
            # Add plural marking if this was a plural form
            if was_plural and vs["noun"] > 0:
                vs = vs.copy()  # Don't modify the original vocabulary vector
                vs["plural"] = 1.0
            
            # Calculate confidence based on compound length
            # Longer compounds get higher confidence when they exist
            if group_size == 1:
                confidence = 0.7  # Base confidence for single words
                description = f"single-word({tok})"
            elif group_size == 2:
                confidence = 1.0  # Higher confidence for two-word compounds
                description = f"two-word-compound({tok})"
            elif group_size == 3:
                confidence = 1.2  # Highest confidence for three-word compounds
                description = f"three-word-compound({tok})"
            else:
                confidence = 0.5
                description = f"{group_size}-word-compound({tok})"
            
            return vs, confidence, description
        
        # Try verb inflection if single word and not in vocabulary
        elif group_size == 1:
            root_verb, inflection_type, found_root = find_root_verb(tok)
            if found_root:
                vs = vector_from_word(root_verb)
                vs.word = tok.lower()
                if inflection_type:
                    vs[inflection_type] = 1.0
                return vs, 0.6, f"inflected-verb({tok}→{root_verb})"
            
            # Try adjective inflection (comparative/superlative)
            from latn.utils.adjective_inflector import find_root_adjective
            base_adj, inflection_type, found_adjective = find_root_adjective(tok)
            if found_adjective:
                try:
                    vs = vector_from_word(base_adj)
                    if vs and vs["adj"] > 0:  # Only if base is actually an adjective
                        vs.word = tok.lower()
                        if inflection_type:
                            vs[inflection_type] = 1.0
                            # Boost host-registered semantic features without
                            # baking a domain ontology into LATN.
                            multiplier = 1.2 if inflection_type == 'comp' else 1.5
                            from latn.An_N_Space_Model.vector_dimensions import SEMANTIC_DIMENSIONS
                            for key in SEMANTIC_DIMENSIONS:
                                if vs[key] != 0:
                                    vs[key] = vs[key] * multiplier
                        return vs, 0.7, f"inflected-adjective({tok}→{base_adj})"
                except ValueError:
                    # Base adjective not in vocabulary - fall through to unknown token
                    pass
            
            # Single word not found in vocabulary - create <unknown> token
            vs = vector_from_features("unknown")
            vs.word = tok.lower()
            return vs, 0.1, f"unknown({tok})"
        
        # Multi-word compound not found in vocabulary - don't create unknown token
        return None, 0, ""


def latn_tokenize_best(sentence):
    """
    LATN Tokenizer: Returns the best hypothesis as a list of VectorSpace tokens.
    
    This provides backward compatibility with the original tokenize() function
    while using multi-hypothesis tokenization internally.
    """
    hypotheses = latn_tokenize_layer1(sentence)
    if hypotheses:
        return hypotheses[0].tokens
    else:
        # If no hypotheses, return empty list (shouldn't happen with proper vocabulary)
        return []
