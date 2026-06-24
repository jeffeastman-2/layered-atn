#!/usr/bin/env python3
"""
LATN Layer 2 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 2 NounPhrase tokens.
It bridges between parsed NounPhrase structures and scene objects.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import copy
from itertools import product

from latn.pos.noun_phrase import NounPhrase
from latn.lexer.scene_adapter import SceneAdapter, GroundedEntity
from latn.lexer.hypothesis import TokenizationHypothesis


@dataclass
class Layer2GroundingResult:
    """Result of Layer 2 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_objects: List[GroundedEntity] = None  # Changed to list to support plural pronouns
    grounded_noun_phrase: Optional[NounPhrase] = None  # The NP with grounding info added
    description: str = ""
    alternative_matches: List[Tuple[float, GroundedEntity]] = None
    
    def __post_init__(self):
        if self.resolved_objects is None:
            self.resolved_objects = []
        if self.alternative_matches is None:
            self.alternative_matches = []

class GroundingOption:
    original_np: NounPhrase
    confidence: float
    resolved_objects: Optional[List[GroundedEntity]]
    grounded_np: Optional[NounPhrase]    

    def __init__(self, original_np=None, confidence=0.0, resolved_objects=None, grounded_np=None):
        self.original_np = original_np
        self.confidence = confidence
        self.resolved_objects = resolved_objects
        self.grounded_np = grounded_np

class Layer2SemanticGrounder:
    """Semantic grounding for LATN Layer 2 NounPhrase tokens."""
    
    def __init__(self, scene_model: SceneAdapter):
        self.scene_model = scene_model

    def ground_pronoun(self, np: NounPhrase) -> Layer2GroundingResult:
        """Ground a pronoun NounPhrase using specialized pronoun resolution.
        
        Args:
            np: The pronoun NounPhrase to ground
            
        Returns:
            Layer2GroundingResult with resolved object(s) and confidence information
        """
        try:
            resolved_objects = self.scene_model.resolve_pronoun(np.pronoun)
            if not resolved_objects:
                return Layer2GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"No objects found for pronoun '{np.pronoun}'"
                )
            alternative_matches = []
            # Handle singular vs plural pronouns based on grammatical features
            if np.vector.isa("singular"):
                # Singular pronouns: take the most recent/relevant single object
                resolved_object_list = [resolved_objects[0]]  # Single object in list
                if len(resolved_objects) > 1:
                    alternative_matches = [(1.0, obj) for obj in resolved_objects[1:]]
                confidence = 1.0  # High confidence for singular reference
                description = f"Resolved singular pronoun '{np.pronoun}' to {resolved_objects[0].entity_id}"
                pronoun_type = 'singular'
            elif np.vector.isa("plural"):
                # Plural pronouns: ground to all resolved objects equally
                resolved_object_list = resolved_objects  # All objects
                confidence = 1.0  # High confidence for plural reference to known objects
                description = f"Resolved plural pronoun '{np.pronoun}' to {len(resolved_objects)} entities: {[obj.entity_id for obj in resolved_objects]}"
                pronoun_type = 'plural'
            else:
                # Unknown grammatical number - fallback
                resolved_object_list = [resolved_objects[0]]  # Single object fallback
                if len(resolved_objects) > 1:
                    alternative_matches = [(1.0, obj) for obj in resolved_objects[1:]]
                confidence = 0.5
                description = f"Resolved pronoun '{np.pronoun}' (unknown number) to {resolved_objects[0].entity_id}"
                pronoun_type = 'unknown'
            
            # Add grounding information to the NounPhrase
            grounded_np = copy.deepcopy(np)
            grounding_info = {
                'scene_objects': resolved_object_list,  # Store all resolved objects
                'confidence': confidence,
                'type': 'pronoun_resolution',
                'pronoun_type': pronoun_type
            }
            grounded_np.grounding = grounding_info               
            return Layer2GroundingResult(
                success=True,
                confidence=confidence,
                resolved_objects=resolved_object_list,  
                grounded_noun_phrase=grounded_np,
                description=description,
                alternative_matches=alternative_matches  
            )                
        except ValueError as e:
            return Layer2GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Pronoun resolution failed: {e}"
            )
    
    def ground_np(self, np: NounPhrase) -> Layer2GroundingResult:
        """Ground a NounPhrase to scene objects using SceneModel.        
        Args:
            np: The NounPhrase to ground            
        Returns:
            Layer2GroundingResult with resolved object(s) and confidence information
        """
        if not isinstance(np, NounPhrase):
            return Layer2GroundingResult(
                success=False,
                confidence=0.0,
                description=f"Expected NounPhrase, got {type(np).__name__}"
            )        
        # Handle pronoun NPs using specialized pronoun resolution
        if np.vector.isa("pronoun"):
            return self.ground_pronoun(np)
        else:
            # Handle regular NPs using SceneModel - get all matching objects
            candidates = self.scene_model.resolve_noun_phrase(np)
            if not candidates:
                return Layer2GroundingResult(
                    success=False,
                    confidence=0.0,
                    description=f"No scene objects match NP: {np}"
                )        
            alternative_matches = []
            # Determine if this NP should ground to multiple objects
            if np.vector.isa("plural"):
                # Ground to ALL matching objects (e.g., "all the red objects", "the cubes,")
                resolved_object_list = [obj for conf, obj in candidates]
                avg_confidence = sum(conf for conf, obj in candidates) / len(candidates)
                object_ids = [obj.object_id for obj in resolved_object_list]
                description = f"Grounded NP '{np}' to {len(resolved_object_list)} objects: {object_ids}"
            else:
                # Ground to the best single match (e.g., "a cube", "a red object")
                best_confidence, best_object = candidates[0]
                resolved_object_list = [best_object]
                avg_confidence = best_confidence
                description = f"Grounded NP '{np}' to {best_object.object_id}"
                # store alternative matches
                if len(candidates) > 1:
                    alternative_matches = candidates[1:]  # Store alternatives excluding best match            
            # Add grounding information directly to the NounPhrase
            grounded_np = copy.deepcopy(np)
            grounding_info = {
                'scene_objects': resolved_object_list,  # Store all resolved objects
                'confidence': avg_confidence,
                'type': 'scene_object',
                'multiple_objects': np.vector.isa("plural")
            }           
            grounded_np.grounding = grounding_info            
            return Layer2GroundingResult(
                success=True,
                confidence=avg_confidence,
                resolved_objects=resolved_object_list,
                grounded_noun_phrase=grounded_np,
                description=description,
                alternative_matches=alternative_matches
            )
    
    def extract_noun_phrases(self, layer2_hypotheses: List[TokenizationHypothesis]) -> List[NounPhrase]:
        """Extract NounPhrase objects from Layer 2 processing.
        
        The token stream is the single source of truth - NP tokens are already
        properly placed in the tokens list by replace_np_sequences().
        
        Clean semantics:
        - If grounding was enabled: Return SceneObjectPhrase for grounded NPs + NounPhrase for unbound NPs
        - If grounding was disabled: Return only NounPhrase objects (all NPs)
        """
        noun_phrases = []
        
        for i, hypothesis in enumerate(layer2_hypotheses):
            # Look for NP tokens in the hypothesis token stream
            for j, token in enumerate(hypothesis.tokens):
                if token.isa("NP"):
                    if token.isa("conj"):
                        conj = token.phrase
                        for np in conj.phrases:
                            noun_phrases.append(np)
                    else:
                        noun_phrases.append(token.phrase)
            
        return noun_phrases
    
    def ground_layer2(self, layer2_hypotheses: List[TokenizationHypothesis]) -> Tuple[List[TokenizationHypothesis], List[Layer2GroundingResult]]:
        """Multiply hypotheses based on grounding results using two-pass algorithm."""
        all_grounded_hypotheses = []
        all_grounding_results = []
        
        for hypothesis in layer2_hypotheses:
            # Pass 1: Collect all potential groundings for each NP in this hypothesis
            np_grounding_options = []
            # this will handle conjoined NPs as well
            nps = self.extract_noun_phrases([hypothesis])
            for original_np in nps: 
                # Get all possible groundings for this NP
                grounding_result = self.ground_np(original_np)
                all_grounding_results.append(grounding_result)
                if grounding_result.success:
                    # grounded_np = copy.deepcopy(original_np)
                    grounded_np = grounding_result.grounded_noun_phrase
                    if grounding_result.alternative_matches:
                        # Include best match + alternatives
                        objs = grounding_result.resolved_objects
                        grounding_option = GroundingOption(
                            original_np=original_np, 
                            confidence=grounding_result.confidence, 
                            resolved_objects=objs, 
                            grounded_np=grounded_np)
                        options = [grounding_option]
                        for conf, alt_obj in grounding_result.alternative_matches:
                            # Create alternative grounded NounPhrase
                            alt_np = copy.deepcopy(grounded_np)
                            alt_np.grounding['scene_objects'] = [alt_obj]
                            alt_np.grounding['confidence'] = conf
                            alt_grounding_option = GroundingOption(
                                original_np=original_np, 
                                confidence=conf, 
                                resolved_objects=[alt_obj], 
                                grounded_np=alt_np)
                            options.append(alt_grounding_option)
                        np_grounding_options.append(options)
                    else:
                        # Single best match
                        np_grounding_options.append([GroundingOption(
                            original_np=original_np, 
                            confidence=grounding_result.confidence, 
                            resolved_objects=grounding_result.resolved_objects, 
                            grounded_np=grounded_np)])
                else:
                    # No grounding found
                    np_grounding_options.append([GroundingOption(
                        original_np=original_np, 
                        confidence=0.5, 
                        resolved_objects=[], 
                        grounded_np=None)])

            # Pass 2: Generate combinatorial hypotheses
            if np_grounding_options:
                for combo in product(*np_grounding_options):
                    new_hypothesis = copy.deepcopy(hypothesis)                    
                    combo_confidence = 1.0
                    grounding_confidences = []
                    for grounding_option in combo:
                        for orig_token, new_token in zip(hypothesis.tokens, new_hypothesis.tokens):
                            if orig_token.isa("conj") and orig_token.isa("NP"):
                                # Handle conjoined NPs
                                conj = orig_token.phrase
                                for idx, np in enumerate(conj.phrases):
                                    if np == grounding_option.original_np:
                                        # Replace with grounded NP (if any)
                                        if grounding_option.grounded_np:
                                            new_conj = new_token.phrase
                                            new_conj.phrases[idx] = grounding_option.grounded_np
                                        break
                            elif orig_token.isa("NP") and orig_token.phrase == grounding_option.original_np:
                                # Replace new token's phrase with grounded NP (if any)
                                if grounding_option.grounded_np:
                                    new_token.phrase = grounding_option.grounded_np
                                    #new_token.attributes.update(grounding_option.grounded_np.vector.features)
                                    break
                                else:
                                    continue
                        grounding_confidences.append(grounding_option.confidence)

                    # ← ADD THIS: Calculate combo_confidence from grounding results
                    if grounding_confidences:
                        # Option 1: Average grounding confidence with softer penalty
                        avg_grounding = sum(grounding_confidences) / len(grounding_confidences)
                        combo_confidence = 0.7 + 0.3 * avg_grounding  # Base 70% + 30% from grounding
                        
                        # Option 2: Multiplicative (more conservative)
                        # combo_confidence = 1.0
                        # for conf in grounding_confidences:
                        #     combo_confidence *= (0.8 + 0.2 * conf)  # Softer multiplicative penalty
                       
                    # Update hypothesis confidence
                    new_hypothesis.confidence = hypothesis.confidence * combo_confidence
                    all_grounded_hypotheses.append(new_hypothesis)
            else:
                # No NPs to ground - keep original hypothesis
                hypothesis.grounding_results = []  # ← Empty list for consistency
                all_grounded_hypotheses.append(hypothesis)
        
        # Sort hypotheses by confidence
        all_grounded_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return all_grounded_hypotheses, all_grounding_results