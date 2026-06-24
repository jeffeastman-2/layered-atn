#!/usr/bin/env python3
"""
LATN Layer 4 Semantic Grounding

This module provides semantic grounding for LATN Layer 4 VerbPhrase tokens.
The per-VP validity verdict is delegated to the active VPGroundingPolicy
(latn.lexer.vp_policy); this module owns only hypothesis iteration. It
does NOT execute actions or modify the scene.
"""

from typing import List
from dataclasses import dataclass

from latn.lexer.scene_adapter import SceneAdapter
from latn.lexer.hypothesis import TokenizationHypothesis
from latn.lexer.vp_policy import get_active_vp_policy


@dataclass
class Layer4GroundingResult:
    """Result of Layer 4 semantic grounding operation."""
    success: bool
    confidence: float
    description: str = ""


class Layer4SemanticGrounder:
    """Semantic grounding for LATN Layer 4 VerbPhrase tokens.

    Layer 4 understands the meaning/context of verb phrases; it does not
    execute actions. The per-VP validity rule is supplied by the active
    VPGroundingPolicy so non-Engraf domains are not fail-closed-rejected.
    """

    def __init__(self, scene_model: SceneAdapter):
        self.scene_model = scene_model

    def ground_layer4(self, hypotheses: List[TokenizationHypothesis]) -> List[TokenizationHypothesis]:
        """Eliminate hypotheses whose verb phrases are invalid under the
        active VPGroundingPolicy. Does not execute actions.

        Args:
            hypotheses: List of TokenizationHypothesis objects to ground

        Returns:
            Processed hypotheses with validated VP semantic grounding
        """
        policy = get_active_vp_policy()
        result_hypotheses = []
        for hyp in hypotheses:
            # Validate the verb phrases in the hypothesis (can have >= 0)
            valid = True
            for token in hyp.tokens:
                if token.isa("VP"):
                    if token.phrase:  # Ensure phrase exists
                        if token.isa("conj"):
                            vps = token.phrase.phrases
                            for vp in vps:  # type: ignore
                                if not policy.validate_vp(vp):
                                    valid = False
                                    break
                        else:
                            if not policy.validate_vp(token.phrase):
                                valid = False
                                break
                elif token.isa("NP"):
                    continue
                elif token.isa("conj"):
                    continue
                else:
                    break
            if valid:
                result_hypotheses.append(hyp)
        return result_hypotheses
