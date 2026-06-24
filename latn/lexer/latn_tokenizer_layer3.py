#!/usr/bin/env python3
"""
LATN Layer 3: PrepositionalPhrase Token Replacement

This module implements Layer 3 of the LATN (Layered Augmented Transition Network) system.
Layer 3 replaces prepositional phrase constructions with single PrepositionalPhrase tokens.

Layer 3 builds on:
- Layer 1: Multi-hypothesis tokenization with morphological inflection

This layer identifies prepositional phrases like "in the red box", "on a very large sphere", "[1,2,3]"
and replaces them with single PrepositionalPhrase tokens containing the semantic meaning.
"""

from typing import List

from latn.lexer.hypothesis import TokenizationHypothesis  

def generate_pp_attachment_combinations(layer3_hypotheses):
    """Generate all possible PP attachment combinations."""
    from copy import deepcopy
    from itertools import product
    
    all_combinations = []
    
    for hypothesis in layer3_hypotheses:
        # Find PP tokens and their possible attachment targets
        pp_positions = []
        attachment_options = []
        
        for i, token in enumerate(hypothesis.tokens):
            if token.isa("PP"):
                pp_positions.append(i)
                
                # Find all preceding NP/PP tokens as potential attachment targets
                targets = [None]  # None = no attachment
                for j in range(i):
                    prev_token = hypothesis.tokens[j]
                    if prev_token.isa("NP") or prev_token.isa("PP"):
                        targets.append(j)

                attachment_options.append(targets)  # None = no attachment

        if not pp_positions:
            # No PPs to attach, keep original hypothesis
            all_combinations.append(hypothesis)
            continue
        
        # Generate cartesian product of all attachment combinations
        for combination in product(*attachment_options):
            # Create new hypothesis with this attachment combination
            new_hypothesis = deepcopy(hypothesis)
            
            # Track which PP tokens will be removed (those that attach to something)
            tokens_to_remove = set()
            
            # Add attachment references and mark for removal if attached
            for pp_idx, target_idx in zip(pp_positions, combination):
                pp_token = new_hypothesis.tokens[pp_idx]                
                if target_idx is not None:  # PP attaches to something
                    target_token = new_hypothesis.tokens[target_idx]
                    # Handle attachment to a NP
                    if target_token.isa("NP"):
                        np_obj = target_token.phrase
                        np_obj.add_prepositional_phrase(pp_token.phrase)
                    # Handle attachment to a PP (attach to its NP)
                    elif target_token.isa("PP"):
                        pp_obj = target_token.phrase
                        np_obj = pp_obj.noun_phrase
                        np_obj.add_prepositional_phrase(pp_token.phrase)
                    # Remove the PP token since it's now bound for identification
                    tokens_to_remove.add(pp_idx)
            
            # Remove attached PP tokens from the token sequence
            new_hypothesis.tokens = [token for i, token in enumerate(new_hypothesis.tokens)
                                   if i not in tokens_to_remove]

            # Update confidence based on token complexity
            num_tokens = len(new_hypothesis.tokens)
            new_hypothesis.confidence =  hypothesis.confidence / num_tokens if num_tokens > 0 else hypothesis.confidence
                        
            all_combinations.append(new_hypothesis)
    
    return all_combinations
