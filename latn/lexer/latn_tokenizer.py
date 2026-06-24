#!/usr/bin/env python3
"""
LATN generic tokenizer for all layers

This module implements all layers of the LATN (Layered Augmented Transition Network) system.
Layer 2 replaces noun phrase constructions with single NounPhrase tokens.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.
Layer 4 replaces verb phrase constructions with single VerbPhrase tokens.
Layer 5 replaces sentence phrase constructions with single SentencePhrase tokens.

Layer 2 builds on:
- LayerTokenizer: Multi-hypothesis tokenization with morphological inflection
"""

from typing import List

from latn.lexer.hypothesis import TokenizationHypothesis  
from latn.lexer.token_stream import TokenStream
from latn.atn.core import run_atn
from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.lexer.vector_space import VectorSpace

class LATNLayerTokenizer:
    """Base class for LATN Layer Tokenizers (Layer 2, 3, 4, 5)"""

    def __init__(self, layer: int, atn_builder, nonterminal_type_builder):
        self.layer = layer  # Layer number (2, 3, 4, or 5)
        self.atn_builder = atn_builder  # Function to build the appropriate ATN
        self.nonterminal_builder = nonterminal_type_builder    # Function to build nonterminal pose.g., "NP", "PP", "VP", "SP"

    def _create_nonterminal_token(self, nonterminal_or_conj) -> VectorSpace:
        """Create a token from a parsed Phrase or ConjunctionPhrase object.
        This creates a single token that represents the entire nonterminal phrase or 
        coordinated phrase construction.
        """
        # Create a new token with the semantic vector
        token = VectorSpace()
        
        # Handle ConjunctionPhrase (coordinated phrases)
        if isinstance(nonterminal_or_conj, ConjunctionPhrase):
            # Copy the semantic content from the ConjunctionPhrase
            if hasattr(nonterminal_or_conj, 'vector') and nonterminal_or_conj.vector:
                from latn.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                for dim in VECTOR_DIMENSIONS:
                    value = nonterminal_or_conj.vector[dim]
                    if value != 0.0:
                        token[dim] = value
            
            # Mark this as a ConjunctionPhrase token
            token["conj"] = 1.0
            token["plural"] = 1.0
            token[self.nonterminal_builder.phrase_type()] = 1.0  # Also mark as nonterminal_type-phrase since it functions as one
            
            # Create descriptive word - use phrase-level display if available
            if hasattr(nonterminal_or_conj, '_phrase_level_display'):
                token.word = nonterminal_or_conj._phrase_level_display
            elif hasattr(nonterminal_or_conj, 'get_original_text'):
                token.word = f"CONJ-{self.nonterminal_builder.phrase_type()}({nonterminal_or_conj.get_original_text()})"
            else:
                token.word = f"CONJ-{self.nonterminal_builder.phrase_type()}"
        
        # Handle regular Phrase
        else:
            # Copy the semantic content from the Phrase
            if hasattr(nonterminal_or_conj, 'vector') and nonterminal_or_conj.vector:
                from latn.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                for dim in VECTOR_DIMENSIONS:
                    value = nonterminal_or_conj.vector[dim]
                    if value != 0.0:
                        token[dim] = value
            
            # Mark this as a Phrase token
            token["singular"] = 1.0
            token[self.nonterminal_builder.phrase_type()] = 1.0
            
            # Create descriptive word
            token.word = nonterminal_or_conj.descriptive_word()

        # Store reference to original object for Layer 3
        token.phrase = nonterminal_or_conj
        
        return token


    def find_nonterminal_sequences(self, tokens: List[VectorSpace]) -> List[tuple]:
        """Find nonterminal phrase sequences in a list of tokens using the supplied ATN.
        
        Returns list of (start_idx, end_idx, nonterminal_object) tuples.
        Uses greedy left-to-right parsing: try ATN at each position, if successful
        consume those tokens and continue from the next position.
        """
        nt_sequences = []
        i = 0
        
        while i < len(tokens):
            # Use TokenStream position tracking to determine how many tokens were consumed
            subsequence = tokens[i:]  # Use all remaining tokens
            best_nt = None
            best_end = i
            
            # First, try to parse a simple nonterminal phrase
            try:
                ts = TokenStream(subsequence)
                nt = self.nonterminal_builder()
                nt_start, nt_end = self.atn_builder(nt, ts)
                result = run_atn(nt_start, nt_end, ts, nt)

                if result is not None:
                    # Found a valid simple nonterminal phrase
                    best_nt = result
                    best_end = i + ts.position - 1
                    
                    # Check for conjunctions to build coordinated phrases
                    while ts.peek() and (ts.peek().isa("conj") or ts.peek().isa("comma")):
                        # There's a conjunction! Try to parse another nonterminal phrase
                        conj_token = ts.next()  # consume the conjunction
                        while conj_token.isa("comma") and ts.peek().isa("conj"):
                            conj_token = ts.next()  # consume the conjunction after comma
                        nt = self.nonterminal_builder()
                        nt_start, nt_end = self.atn_builder(nt, ts)
                        nt2_result = run_atn(nt_start, nt_end, ts, nt)
                        
                        if nt2_result is not None:
                            # Successfully parsed another nonterminal - create/extend coordination
                            if isinstance(best_nt, type(nt)):
                                # Convert to ConjunctionPhrase
                                coord_np = ConjunctionPhrase(conj_token, phrases=[best_nt, nt2_result])
                                coord_np.vector["plural"] = 1.0
                                best_nt = coord_np
                            elif isinstance(best_nt, ConjunctionPhrase):
                                if best_nt.vector.isa("comma"):
                                    best_nt.vector["comma"] = 0.0
                                    best_nt.vector += conj_token
                                if (best_nt.vector.isa("and") and conj_token.isa("or")) \
                                    or (best_nt.vector.isa("or") and conj_token.isa("and")):
                                    raise ValueError("Mixed conjunctions 'and' and 'or' not supported in coordination")
                                best_nt.phrases.append(nt2_result)
                            
                            # Update best_end to include the newly parsed nonterminal
                            best_end = i + ts.position - 1
                        else:
                            # Failed to parse second nonterminal - break out of coordination loop
                            # Put the conjunction token back by rewinding
                            ts.position -= 1
                            break
            except Exception:
                    # Simple nonterminal parsing also failed
                    pass
            
            if best_nt is not None:
                # Found a nonterminal, add it and skip past it
                nt_sequences.append((i, best_end, best_nt))
                i = best_end + 1
            else:
                # No nonterminal found starting at position i, move to next position
                i += 1
        
        return nt_sequences


    def replace_nt_sequences(self, tokens: List[VectorSpace], nt_sequences: List[tuple]) -> List[VectorSpace]:
        """Replace NT sequences with NT tokens."""
        if not nt_sequences:
            return tokens
        
        new_tokens = []
        i = 0
        
        for start_idx, end_idx, nt in nt_sequences:
            # Add tokens before this NT
            while i < start_idx:
                new_tokens.append(tokens[i])
                i += 1
            
            # Add the NP token
            np_token = self._create_nonterminal_token(nt)
            new_tokens.append(np_token)
            
            # Skip the original NT tokens
            i = end_idx + 1
        
        # Add remaining tokens
        while i < len(tokens):
            new_tokens.append(tokens[i])
            i += 1
        
        return new_tokens


    def find_coordination_hypotheses(self, tokens: List[VectorSpace]) -> List[List[tuple]]:
        """Generate multiple coordination hypotheses for ambiguous structures.
        
        For sentences with coordination ambiguity, this generates alternative
        interpretations that can be disambiguated by higher layers.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            List of hypothesis alternatives, each containing NP sequences
        """
        hypotheses = []
        sequences = self.find_nonterminal_sequences(tokens)
        hypotheses.append(sequences)
        
        return hypotheses

    def latn_tokenize_layer(self, previous_layer_hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
        """LATN Tokenizer: Replace nonterminal phrase sequences with nonterminal phrase tokens.
        
        This is the main entry point for all layer tokenizations. It takes previous layer
        hypotheses and identifies new phrase constructions, replacing them with 
        single phrase tokens.
        
        Args:
            previous_layer_hypotheses: List of TokenizationHypothesis from previous layer
            
        Returns:
            List of TokenizationHypothesis objects, ranked by confidence
        """
        layer_hypotheses = []
        
        for base_hyp in previous_layer_hypotheses:      
            # Generate multiple coordination hypotheses for ambiguous structures
            coordination_hypotheses = self.find_coordination_hypotheses(base_hyp.tokens)
            for i, nt_sequences in enumerate(coordination_hypotheses):
                if nt_sequences:
                    # Create hypothesis with NP replacements
                    new_tokens = self.replace_nt_sequences(base_hyp.tokens, nt_sequences)
                    new_confidence = base_hyp.confidence                
                    # Create description for this hypothesis
                    description = f"Layer {self.layer}: {len(nt_sequences)} nonterminal sequences"
                    
                    layer_hyp = TokenizationHypothesis(
                        tokens=new_tokens,
                        confidence=new_confidence,
                        description=description,
                        replacements=[(start, end, self._create_nonterminal_token(nt)) for start, end, nt in nt_sequences]
                    )
                    layer_hypotheses.append(layer_hyp)
                else:
                    # No nonterminal sequences found, return base hypothesis
                    layer_hyp = TokenizationHypothesis(
                        tokens=base_hyp.tokens,
                        confidence=base_hyp.confidence,
                        description=f"Layer {self.layer}: No nonterminal sequences found",
                        replacements=[]
                    )
                    layer_hypotheses.append(layer_hyp)
        
        # Sort by confidence (highest first)
        #layer_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return layer_hypotheses
