#!/usr/bin/env python3
"""
LATN Layer 5 Semantic Grounding

This module provides semantic grounding and execution capabilities for LATN Layer 5 
Sentence tokens. It takes well-formed sentences and executes them in the scene.
"""

from typing import List
from dataclasses import dataclass

from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.pos.sentence_phrase import SentencePhrase
from latn.lexer.scene_adapter import SceneAdapter
from latn.lexer.hypothesis import TokenizationHypothesis


@dataclass
class Layer5GroundingResult:
    """Result of Layer 5 semantic grounding operation."""
    success: bool
    confidence: float
    executed_actions: List[str]
    scene_changes: List[str]
    description: str = ""


class Layer5SemanticGrounder:
    """Semantic grounding and execution for LATN Layer 5 Sentence tokens.

    Layer 5 evaluates the hypotheses to reject those that are not well-formed
    or do not align with the current scene context.
    """

    def __init__(self, scene_model: SceneAdapter):
        self.scene_model = scene_model
        
    
    def ground_layer5(self, hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
        """Apply grounding to hypotheses following the pattern of other layers.
        
        Args:
            hypotheses: Input hypotheses to ground

        Returns:
            List of grounded hypotheses.
        """
        if not hypotheses:
            return []

        from latn.lexer.sp_policy import get_active_sp_policy
        policy = get_active_sp_policy()
        grounded_hypotheses = []

        for hypothesis in hypotheses:
            # Extract the sentence phrases this hypothesis produced, and note
            # whether EVERY token folded into one (no leftover non-SP tokens).
            # The keep/reject verdict belongs to the active SP policy:
            # StrictSPPolicy demands full coverage (strict, default);
            # PermissiveSPPolicy keeps any hypothesis with >= 1 SP.
            sentence_phrases = []
            all_tokens_are_sp = True
            for token in hypothesis.tokens:
                sp = token.phrase if hasattr(token, 'phrase') else None
                if sp and isinstance(sp, SentencePhrase):
                    sentence_phrases.append(sp)
                elif sp and isinstance(sp, ConjunctionPhrase) and \
                        all(isinstance(p, SentencePhrase) for p in sp.phrases):
                    sentence_phrases.extend(sp.phrases)
                else:
                    all_tokens_are_sp = False
            if policy.accept_hypothesis(sentence_phrases, all_tokens_are_sp):
                grounded_hypotheses.append(hypothesis)

        return grounded_hypotheses

    def extract_sentence_phrases(self, hypothesis: TokenizationHypothesis) -> List[SentencePhrase]:
        """Extract sentence phrases from a tokenization hypothesis.

        Args:
            hypothesis: The tokenization hypothesis to extract from.

        Returns:
            List of extracted SentencePhrase objects.
        """
        sentence_phrases = []
        for token in hypothesis.tokens:
            sp = token.phrase if hasattr(token, 'phrase') else None
            if sp and isinstance(sp, SentencePhrase):
                sentence_phrases.append(sp)
            elif sp and isinstance(sp, ConjunctionPhrase):
                for part in sp.phrases:
                    if isinstance(part, SentencePhrase):
                        sentence_phrases.append(part)
        return sentence_phrases