#!/usr/bin/env python3
"""
LATN Layer 3 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 3 PrepositionalPhrase tokens.
It bridges between parsed PrepositionalPhrase structures and spatial locations/relationships.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
 
from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.pos.prepositional_phrase import PrepositionalPhrase
from latn.lexer.vector_space import VectorSpace
from latn.lexer.hypothesis import TokenizationHypothesis
from latn.utils.debug import debug_print
from latn.utils.spatial_validation import SpatialValidator
from latn.lexer.spatial_policy import get_active_spatial_policy
from latn.lexer.scene_adapter import SceneAdapter


@dataclass
class Layer3GroundingResult:
    """Result of Layer 3 semantic grounding operation."""
    success: bool
    confidence: float
    resolved_object: Optional[VectorSpace] = None
    description: str = ""
    alternative_matches: List[Tuple[float, VectorSpace]] = None

    def __post_init__(self):
        if self.alternative_matches is None:
            self.alternative_matches = []


class Layer3SemanticGrounder:
    """Semantic grounding for LATN Layer 3 PrepositionalPhrase tokens."""
    
    def __init__(self, scene_model: SceneAdapter):
        self.scene_model = scene_model
    
    def ground_layer3(self, layer3_hypotheses) -> List[TokenizationHypothesis]:
        """Two-pass PP attachment resolution with spatial validation and semantic grounding.
        
        Args:
            layer3_hypotheses: Hypotheses with PP tokens to process
            
        Returns:
            Processed hypotheses with validated PP attachments and semantic grounding
        """
        # Spatial validation filter
        validated_hypotheses = self._validate_spatial_attachments(layer3_hypotheses)
        debug_print(f"✅ Pass 2: {len(validated_hypotheses)} spatially valid combinations")        
        return validated_hypotheses

    def _validate_prep_spatial_relationships(self, pp, target_np) -> bool:
        obj1s = target_np.grounding.get('scene_objects') if target_np.grounding else None
        pp_np = pp.noun_phrase
        policy = get_active_spatial_policy()
        if not policy.applies_to(pp.vector):
            return True  # This relationship is interpreted downstream.
        obj2s = pp_np.grounding.get('scene_objects') if pp_np.grounding else None
        if obj1s is None or obj2s is None:
            debug_print(f"❌ Cannot validate spatial relationship: missing grounding")
            return False
        matches = SpatialValidator.validate_spatial_relationship(pp.vector, obj1s, obj2s)
        debug_print(f"🔍 Spatial score for '{pp.vector}': {matches}") 
        new_grounding = []
        for match, obj in zip(matches, obj1s):
            if match:   # valid match => keep the object in the grounding
                new_grounding.append(obj)
        if new_grounding:
            target_np.grounding['scene_objects'] = new_grounding
            return True
        else:
            return False  # No valid grounding left

    def validate_grounded_np(self, grounded_np) -> bool:
        """Validate a grounded NP's PP attachments using spatial reasoning."""
        if not grounded_np:
            return False  # No grounding to validate        
        preps = grounded_np.prepositions
        if not preps:
            return True  # No PPs to validate        
        okay = True
        for pp_obj in preps:
            # Validate each PP object using the spatial relationship validation function
            okay = okay and pp_obj.evaluate_boolean_function(lambda pp: self._validate_prep_spatial_relationships(pp, grounded_np))
            if not okay:    # Stop early if any PP fails validation
                break
        return okay
   
    def _validate_spatial_attachments(self, attachment_hypotheses) -> List[TokenizationHypothesis]:
        """Validate PP attachments using spatial reasoning and return valid hypotheses.
            A hypothesis is valid if all its grounded NPs have PPs with valid spatial relationships.
            Should also cull PPs with invalid groundings within them
        """
        validated = []    
        x = 0    
        for hypothesis in attachment_hypotheses:
            x += 1       # for debugging
            okay = True
            for i, token in enumerate(hypothesis.tokens):
                pString = token.phrase.printString() if token.phrase else "None"
                debug_print(f"Evaluating {pString}")
                if token.isa("NP"):
                    okay = okay and token.phrase.evaluate_boolean_function(lambda np: (np.grounding is None) or 
                                                                          (np.grounding is not None and
                                                                        self.validate_grounded_np(np)))
                elif token.isa("PP"):
                    okay = okay and token.phrase.evaluate_boolean_function(lambda pp: (pp.noun_phrase.grounding is None) or
                                                                    (pp.noun_phrase.grounding is not None and
                                                                    self.validate_grounded_np(pp.noun_phrase)))
            if okay:
                num_tokens = len(hypothesis.tokens) 
                hypothesis.confidence = hypothesis.confidence / num_tokens if num_tokens > 0 else hypothesis.confidence
                debug_print(f"✅ Hypothesis {x} spatially valid with score {hypothesis.confidence:.2f}")
                # Merge validated PPs into their target NPs
                validated.append(hypothesis)  # Valid

        # Sort by confidence (best first)
        validated.sort(key=lambda h: h.confidence, reverse=True)
        return validated
    
    def extract_prepositional_phrases(self, layer3_hypotheses: List[TokenizationHypothesis]) -> List[PrepositionalPhrase]:
        """Extract PrepositionalPhrase objects from Layer 3 hypotheses.
        
        Args:
            layer3_hypotheses: List of Layer 3 tokenization hypotheses
            
        Returns:
            List of PrepositionalPhrase objects found in the hypotheses
        """
        prepositional_phrases = []
        
        for hypothesis in layer3_hypotheses:
            for token in hypothesis.tokens:
                if token.phrase is not None and isinstance(token.phrase, PrepositionalPhrase):
                    prepositional_phrases.append(token.phrase)

        return prepositional_phrases
