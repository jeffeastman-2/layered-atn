#!/usr/bin/env python3
"""
LATN Layer Executor

This module provides layered execution entry points for the LATN system.
Each layer automatically executes all prerequisite layers, allowing clean
entry at any layer for testing and processing.

Refactored for clean separation of concerns:
- Tokenization logic → respective tokenizer modules
- Grounding logic → respective grounder modules
- Coordination logic → this executor (lightweight)
"""

from typing import List, Optional
from dataclasses import dataclass, field

from latn.atn.core import run_atn
from latn.atn.np import build_np_atn
from latn.atn.pp import build_pp_atn
from latn.atn.sentence import build_sentence_atn
from latn.atn.sq import build_sq_atn
from latn.atn.vp import build_vp_atn
from latn.lexer.token_stream import TokenStream
from latn.pos.question_phrase import QuestionPhrase
from latn.lexer.latn_tokenizer import LATNLayerTokenizer
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1, TokenizationHypothesis
from latn.lexer.latn_tokenizer_layer3 import generate_pp_attachment_combinations
from latn.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder, Layer2GroundingResult
from latn.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder, Layer3GroundingResult
from latn.lexer.semantic_grounding_layer4 import Layer4SemanticGrounder, Layer4GroundingResult
from latn.lexer.semantic_grounding_layer5 import Layer5SemanticGrounder, Layer5GroundingResult
from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.lexer.scene_adapter import SceneAdapter
from latn.pos.noun_phrase import NounPhrase
from latn.pos.prepositional_phrase import PrepositionalPhrase
from latn.pos.verb_phrase import VerbPhrase
from latn.pos.sentence_phrase import SentencePhrase


@dataclass
class LayerResult:
    """Result of Layer execution (lexical tokenization)."""
    hypotheses: List[TokenizationHypothesis]
    success: bool
    confidence: float
    description: str = ""

@dataclass
class Layer1Result:
    """Result of Layer 1 execution (lexical tokenization)."""
    hypotheses: List[TokenizationHypothesis]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer2Result:
    """Result of Layer 2 execution (NP tokenization + grounding)."""
    layer1_result: Layer1Result
    hypotheses: List[TokenizationHypothesis]
    noun_phrases: List[NounPhrase]
    grounding_results: List[Layer2GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer3Result:
    """Result of Layer 3 execution (PP tokenization + grounding)."""
    layer2_result: Layer2Result
    hypotheses: List[TokenizationHypothesis]
    prepositional_phrases: List[PrepositionalPhrase]
    grounding_results: List[Layer3GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer4Result:
    """Result of Layer 4 execution (VP tokenization + execution)."""
    layer3_result: Layer3Result
    hypotheses: List[TokenizationHypothesis]
    verb_phrases: List[VerbPhrase]
    grounding_results: List[Layer4GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer5Result:
    """Result of Layer 5 execution (sentence tokenization + execution)."""
    layer4_result: Layer4Result
    hypotheses: List[TokenizationHypothesis]
    sentence_phrases: List[SentencePhrase]
    grounding_results: List[Layer5GroundingResult]
    success: bool
    confidence: float
    description: str = ""
    # Interrogative parses from the parallel SQ pass (empty for statements/
    # commands). The host reads these to answer questions; the declarative SP
    # path above is unaffected either way.
    question_phrases: List["QuestionPhrase"] = field(default_factory=list)


class LATNLayerExecutor:
    """Coordinates execution across all LATN layers with clean delegation to grounders."""
    
    def __init__(self, scene_model: Optional[SceneAdapter] = None):
        self.scene = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model) if scene_model else None
        self.layer3_grounder = Layer3SemanticGrounder(scene_model) if scene_model else None
        self.layer4_grounder = Layer4SemanticGrounder(scene_model) if scene_model else None
        self.layer5_grounder = Layer5SemanticGrounder(scene_model) if scene_model else None

    def enumerate_hypotheses(self, hypotheses: List[TokenizationHypothesis], layer: str = ""):
        """Enumerate and print tokenization hypotheses."""
        for i, hyp in enumerate(hypotheses, 1):
            print(f"  {layer}-{i}  Tokens: {len(hyp.tokens)} | Confidence: {hyp.confidence:.3f}")
            hyp.print_tokens()

    def execute_layer1(self, sentence: str, report=False) -> Layer1Result:
        """Execute Layer 1: Multi-hypothesis lexical tokenization.
        
        Args:
            sentence: Input sentence to tokenize
            report: Whether to generate a report

        Returns:
            LayerResult with tokenization hypotheses
        """
        try:
            hypotheses = latn_tokenize_layer1(sentence)
            
            if not hypotheses or not sentence.strip():
                return Layer1Result(
                    hypotheses=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 2tokenization failed for: '{sentence}'"
                )
            
            # Use confidence from best hypothesis
            best_confidence = hypotheses[0].confidence if hypotheses else 0.0
            if(report):
                print(f"Layer 1 tokenization produced {len(hypotheses)} hypotheses for: '{sentence}'")
                self.enumerate_hypotheses(hypotheses, layer="1t")
            return Layer1Result(
                hypotheses=hypotheses,
                success=True,
                confidence=best_confidence,
                description=f"Layer 1 tokenized '{sentence}' into {len(hypotheses)} hypotheses"
            )
            
        except Exception as e:
            return Layer1Result(
                hypotheses=[],
                success=False,
                confidence=0.0,
                description=f"Layer 1 failed: {e}"
            )
    
    def execute_layer2(self, sentence: str, tokenize_only: bool = False, report: bool = False) -> Layer2Result:
        """Execute Layer 2: NP tokenization (requires Layer 1).
        
        Args:
            sentence: Input sentence to process
            tokenize_only: Whether to only tokenize without grounding
            report: Whether to generate a report

        Returns:
            Layer2Result with NP processing results
        """
        # First execute Layer 1
        layer1_result = self.execute_layer1(sentence, report=report)
        
        if not layer1_result.success:
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=[],
                noun_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 2 failed due to Layer 1 failure: {layer1_result.description}"
            )
        try:
            # Execute Layer 2 NP tokenization
            tokenizer = LATNLayerTokenizer(layer=2, atn_builder=build_np_atn, nonterminal_type_builder=NounPhrase)
            layer2_hypotheses = tokenizer.latn_tokenize_layer(layer1_result.hypotheses)

            if(report):
                print(f"Layer 2 tokenization produced {len(layer2_hypotheses)} hypotheses  for: '{sentence}'")
                self.enumerate_hypotheses(layer2_hypotheses, layer="2t")

            # Semantic grounding with hypothesis multiplication (if enabled)
            if not tokenize_only and self.layer2_grounder:
                grounded_hypotheses, all_grounding_results = self.layer2_grounder.ground_layer2(
                    layer2_hypotheses=layer2_hypotheses
                )
            else:
                # No grounding - keep original hypotheses
                grounded_hypotheses = layer2_hypotheses
                all_grounding_results = []
            
            # Extract noun phrases from the (possibly grounded) hypotheses
            # The tokenizer creates NP tokens with _original_np references, so we can extract them even without grounding
            if self.layer2_grounder:
                all_noun_phrases = self.layer2_grounder.extract_noun_phrases(grounded_hypotheses)
            else:
                # Extract noun phrases directly from tokens when no grounder available
                all_noun_phrases = []
                for hypothesis in grounded_hypotheses:
                    for token in hypothesis.tokens:
                        if token.phrase is not None and token.phrase:
                            all_noun_phrases.append(token.phrase)
            
            # Calculate confidence based on best hypothesis
            layer2_confidence = grounded_hypotheses[0].confidence if grounded_hypotheses else layer1_result.confidence
            overall_confidence = (layer1_result.confidence + layer2_confidence) / 2

            if(report):
                print(f"Layer 2 grounding produced {len(grounded_hypotheses)} hypotheses for: '{sentence}'")
                self.enumerate_hypotheses(grounded_hypotheses, layer="2g")
            
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=grounded_hypotheses,
                noun_phrases=all_noun_phrases,
                grounding_results=all_grounding_results,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 2 processed {len(all_noun_phrases)} noun phrases in {len(grounded_hypotheses)} hypotheses"
            )
            
        except Exception as e:
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=[],
                noun_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 2 failed: {e}"
            )
    
    def execute_layer3(self, sentence: str, tokenize_only: bool = False, report: bool = False) -> Layer3Result:
        """Execute Layer 3: PP tokenization (requires Layer 1-2c).
        
        Args:
            sentence: Input sentence to process
            tokenize_only: Whether to only tokenize without grounding
            report: Whether to generate a report

        Returns:
            Layer3Result with complete LATN processing results
        """
        # First execute Layer 2 (which includes Layer 1-2)
        # Layer 3 spatial validation requires grounded NP tokens from Layer 2
        layer2_result = self.execute_layer2(sentence, report=report)

        if not layer2_result.success:
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=[],
                prepositional_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 3 failed due to Layer 2c failure: {layer2_result.description}"
            )
        
        try:
            # Execute Layer 3 PP tokenization
            tokenizer = LATNLayerTokenizer(layer=3, atn_builder=build_pp_atn, nonterminal_type_builder=PrepositionalPhrase)
            layer3_hypotheses = tokenizer.latn_tokenize_layer(layer2_result.hypotheses)

            # Generate PP attachment combinations to handle multiple PPs
            layer3_hypotheses = generate_pp_attachment_combinations(layer3_hypotheses) 


            if report:
                print(f"Layer 3 tokenization produced {len(layer3_hypotheses)} hypotheses for: '{sentence}'")
                self.enumerate_hypotheses(layer3_hypotheses, layer="3t")

            # Extract PrepositionalPhrase objects
            prepositional_phrases = self.layer3_grounder.extract_prepositional_phrases(layer3_hypotheses) if self.layer3_grounder else []
            
            # Layer 3 grounding - process PP attachments with spatial validation
            if not tokenize_only and self.layer3_grounder:
                # Process PP attachment combinations with spatial validation
                grounded_hypotheses: List[TokenizationHypothesis] = self.layer3_grounder.ground_layer3(layer3_hypotheses)
                if report:
                    print(f"Layer 3 grounding produced {len(grounded_hypotheses)} hypotheses for: '{sentence}'")
                    self.enumerate_hypotheses(grounded_hypotheses, layer="3g")

                # Use the grounded hypotheses as the final result
                final_hypotheses = grounded_hypotheses
                
                # Create grounding results for compatibility (could be empty list)
                grounding_results = []
            else:
                # No grounding - use original hypotheses
                final_hypotheses = layer3_hypotheses
                grounding_results = []
            
            # Calculate confidence
            layer3_confidence = final_hypotheses[0].confidence if final_hypotheses else layer2_result.confidence
            overall_confidence = (layer2_result.confidence + layer3_confidence) / 2
            
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=final_hypotheses,
                prepositional_phrases=prepositional_phrases,
                grounding_results=grounding_results,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 3 processed {len(prepositional_phrases)} prepositional phrases"
            )
            
        except Exception as e:
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=[],
                prepositional_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 3 failed: {e}"
            )

    def execute_layer4(self, sentence: str, tokenize_only: bool = False, report: bool = False) -> Layer4Result:
        """Execute Layer 4: VP tokenization and semantic grounding (requires Layer 1-3).
        
        Args:
            sentence: Input sentence to process
            tokenize_only: Whether to only tokenize without grounding
            report: Whether to generate a report
            
        Returns:
            Layer4Result with complete LATN processing results including verb phrase grounding
        """
        try:
            # Execute Layer 3 first (which includes Layer 1 and 2)
            # Layer 4 should always start from Layer 3 grounding results, not tokenization
            layer3_result = self.execute_layer3(sentence, report=report)
            
            if not layer3_result.success:
                return Layer4Result(
                    layer3_result=layer3_result,
                    hypotheses=[],
                    verb_phrases=[],
                    grounding_results=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 4 failed due to Layer 3 failure: {layer3_result.description}"
                )
            
            # Execute Layer 4 VP tokenization
            tokenizer = LATNLayerTokenizer(layer=4, atn_builder=build_vp_atn, nonterminal_type_builder=VerbPhrase)
            layer4_hypotheses = tokenizer.latn_tokenize_layer(layer3_result.hypotheses)

            if report:
                print(f"Layer 4 tokenization produced {len(layer4_hypotheses)} hypotheses for: '{sentence}'")
                self.enumerate_hypotheses(layer4_hypotheses, layer="4t")

            final_hypotheses: List[TokenizationHypothesis] = []
            grounded_hypotheses: List[TokenizationHypothesis] = []
            # Layer 4 grounding - process VP attachments with semantic validation
            if not tokenize_only and self.layer4_grounder:
                # Process VP attachment combinations with spatial validation
                grounded_hypotheses = self.layer4_grounder.ground_layer4(layer4_hypotheses)
                if report:
                    print(f"Layer 4 grounding produced {len(grounded_hypotheses)} hypotheses for: '{sentence}'")
                # Use the grounded hypotheses as the final result
                final_hypotheses = grounded_hypotheses
            else:
                # No grounding - use original hypotheses
                final_hypotheses = layer4_hypotheses
            
            layer4_confidence = layer4_hypotheses[0].confidence if layer4_hypotheses else layer3_result.confidence
            overall_confidence = (layer3_result.confidence + layer4_confidence) / 2
            verb_phrases = []
            for hyp in final_hypotheses:
                vps = extract_verb_phrases(hyp)
                verb_phrases.extend(vps)
            
            return Layer4Result(
                layer3_result=layer3_result,
                hypotheses=final_hypotheses,
                verb_phrases=verb_phrases,
                grounding_results=[],
                success=True,
                confidence=overall_confidence,
                description=f"Layer 4 processed {len(verb_phrases)} verb phrases"
            )
            
        except Exception as e:
            return Layer4Result(
                layer3_result=layer3_result,
                hypotheses=[],
                verb_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 4 failed: {e}"
            )

    def _parse_questions(self, layer4_hypotheses) -> List[QuestionPhrase]:
        """Run the SQ ATN from the start of each layer-4 hypothesis.

        SQ is a standalone parallel pass: its start state accepts only a
        wh-word or a leading to-be, so a declarative/imperative produces no
        QuestionPhrase and this returns []. Returns one QuestionPhrase per
        hypothesis that parses as a question (best hypothesis first, mirroring
        the layer-4 ranking), so the host can take the top one.
        """
        questions: List[QuestionPhrase] = []
        for hyp in (layer4_hypotheses or []):
            tokens = list(getattr(hyp, "tokens", None) or [])
            if not tokens:
                continue
            ts = TokenStream(tokens)
            q = QuestionPhrase()
            if run_atn(*build_sq_atn(q, ts), ts, q) is not None:
                questions.append(q)
        return questions

    def execute_layer5(self, sentence: str, tokenize_only: bool = False, report: bool = False) -> Layer5Result:
        """Execute Layer 5: Sentence tokenization + execution (requires Layer 1-4).
        
        Args:
            sentence: Input sentence to process
            tokenize_only: Whether to only tokenize without grounding
            report: Whether to generate a report
            
        Returns:
            Layer5Result with complete sentence parsing and execution results
        """
        try:
            # Execute Layer 4 first (which includes Layers 1-3)
            layer4_result = self.execute_layer4(sentence, report=report)

            if not layer4_result.success:
                return Layer5Result(
                    layer4_result=layer4_result,
                    hypotheses=[],
                    sentence_phrases=[],
                    grounding_results=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 5 failed due to Layer 4 failure: {layer4_result.description}"
                )
            
            # Execute Layer 5 sentence tokenization
            tokenizer = LATNLayerTokenizer(layer=5, atn_builder=build_sentence_atn, nonterminal_type_builder=SentencePhrase)
            layer5_hypotheses = tokenizer.latn_tokenize_layer(layer4_result.hypotheses)

            # Parallel interrogative (SQ) pass -- independent of the SP pass
            # above. It only yields a parse when the utterance leads with a
            # wh-word or an inverted to-be, so statements/commands add nothing.
            question_phrases = self._parse_questions(layer4_result.hypotheses)

            if report:
                print(f"Layer 5 tokenization produced {len(layer5_hypotheses)} hypotheses for: '{sentence}'")
                self.enumerate_hypotheses(layer5_hypotheses, layer="5t")

            # Semantic grounding/execution (if enabled)
            if not tokenize_only and self.layer5_grounder:
                grounded_hypotheses = self.layer5_grounder.ground_layer5(layer5_hypotheses)
                if report:
                    print(f"Layer 5 grounding produced {len(grounded_hypotheses)} hypotheses for: '{sentence}'")
                    self.enumerate_hypotheses(grounded_hypotheses, layer="5g")
            else:
                # No grounding - keep original hypotheses
                grounded_hypotheses = layer5_hypotheses
                grounding_results = []
            
            # Extract sentence phrases from grounded hypotheses
            sentence_phrases = []
            if self.layer5_grounder:
                for hyp in grounded_hypotheses:
                    sentence_phrases.extend(self.layer5_grounder.extract_sentence_phrases(hyp))
            layer5_confidence = grounded_hypotheses[0].confidence if grounded_hypotheses else layer4_result.confidence
            overall_confidence = (layer4_result.confidence + layer5_confidence) / 2
            
            description = f"Layer 5 processed {len(sentence_phrases)} sentences"
            
            return Layer5Result(
                layer4_result=layer4_result,
                hypotheses=grounded_hypotheses,
                sentence_phrases=sentence_phrases,
                grounding_results=[],
                success=True,
                confidence=overall_confidence,
                description=description,
                question_phrases=question_phrases,
            )
            
        except Exception as e:
            # Get a valid layer4_result for error case
            if 'layer4_result' not in locals() or layer4_result is None:
                layer4_result = self.execute_layer4(sentence, tokenize_only=True, report=False)
            return Layer5Result(
                layer4_result=layer4_result,
                hypotheses=[],
                sentence_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 5 failed: {e}"
            )

    @property
    def scene_model(self):
        """Compatibility property for tests that expect scene_model attribute."""
        return self.scene
    
    def update_scene_model(self, scene_model: Optional[SceneAdapter]):
        """Update the scene model and reinitialize grounders."""
        self.scene = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model) if scene_model else None
        self.layer3_grounder = Layer3SemanticGrounder(scene_model) if scene_model else None
        self.layer4_grounder = Layer4SemanticGrounder(scene_model) if scene_model else None
        self.layer5_grounder = Layer5SemanticGrounder(scene_model) if scene_model else None
    
    def get_layer_analysis(self, sentence: str, target_layer: int = 3) -> dict:
        """Get detailed analysis of layer execution for debugging/testing."""
        analysis = {
            'input': sentence,
            'target_layer': target_layer
        }
        
        # Execute Layer 1
        layer1_result = self.execute_layer1(sentence)
        analysis['layer1'] = {
            'success': layer1_result.success,
            'confidence': layer1_result.confidence,
            'hypothesis_count': len(layer1_result.hypotheses),
            'description': layer1_result.description
        }

        if target_layer >= 2:
            # Execute Layer 2
            layer2_result = self.execute_layer2(sentence)
            analysis['layer2'] = {
                'success': layer2_result.success,
                'confidence': layer2_result.confidence,
                'noun_phrase_count': len(layer2_result.noun_phrases),
                'grounding_count': len(layer2_result.grounding_results),
                'description': layer2_result.description
            }
        
        if target_layer >= 3:
            # Execute Layer 3
            layer3_result = self.execute_layer3(sentence)
            analysis['layer3'] = {
                'success': layer3_result.success,
                'confidence': layer3_result.confidence,
                'pp_count': len(layer3_result.prepositional_phrases),
                'description': layer3_result.description
            }
        
        return analysis

def extract_sentence_phrases(layer5_hypotheses: List[TokenizationHypothesis]) -> List[SentencePhrase]:
    """Extract SentencePhrase objects from Layer 5 processing.
    
    Args:
        layer5_hypotheses: List of Layer 5 tokenization hypotheses
        
    Returns:
        List of SentencePhrase objects found in the hypotheses
    """
    sentence_phrases = []
    
    for hypothesis in layer5_hypotheses:
        for token in hypothesis.tokens:
            if hasattr(token, '_original_sp') and token.phrase:
                sentence_phrases.append(token.phrase)

    return sentence_phrases

def extract_verb_phrases(hypothesis: TokenizationHypothesis) -> List[VerbPhrase]:
    """Extract VerbPhrase objects from a Layer 4 hypothesis.

    Args:
        layer4_hypothesis: Layer 4 tokenization hypothesis

    Returns:
        List of VerbPhrase objects found in the hypothesis
    """
    verb_phrases = []
    
    for token in hypothesis.tokens:
        vp = token.phrase if hasattr(token, 'phrase') else None
        if vp and isinstance(vp, VerbPhrase):
            verb_phrases.append(vp)
        elif vp and isinstance(vp, ConjunctionPhrase):
            # ConjunctionPhrase.phrases is a List, safe to iterate
            if hasattr(vp, 'phrases') and vp.phrases:
                for part in vp.phrases:  # type: ignore
                    if isinstance(part, VerbPhrase):
                        verb_phrases.append(part)
    return verb_phrases

def tokenize_best(sentence):
    """Tokenize a sentence using the current vocabulary."""
    hypotheses = tokenize_all(sentence)
    hypothesis = hypotheses[0] if len(hypotheses) > 0 else None
    return hypothesis

def tokenize_all(sentence):
    """Tokenize a sentence using LATN Layer 1, returning multiple hypotheses."""
    executor = LATNLayerExecutor()
    result = executor.execute_layer1(sentence)
    assert result is not None, "LATN Layer 1 tokenization failed"
    return result.hypotheses

def is_different_phrase_sequence(a, b):
    if len(a) != len(b):
        return True
    for i in range(len(a)):
        # Compare start_idx and end_idx (indices 0 and 1)
        if a[i][0] != b[i][0] or a[i][1] != b[i][1]:
            return True
        
        # For POS objects (index 2), compare by content rather than identity
        pos_a, pos_b = a[i][2], b[i][2]
        if not pos_a.equals(pos_b):
            return True              
    return False
