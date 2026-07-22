"""The Layer-4 core delegates its verdict without knowing host intents."""

from latn.lexer.hypothesis import TokenizationHypothesis
from latn.lexer.semantic_grounding_layer4 import Layer4SemanticGrounder
from latn.lexer.vector_space import vector_from_features
from latn.lexer.vp_policy import (
    PermissiveVPPolicy,
    VPGroundingPolicy,
    set_active_vp_policy,
    use_vp_policy,
)
from latn.pos.verb_phrase import VerbPhrase


class RejectVPPolicy:
    def validate_vp(self, vp):
        return False


def _hypothesis():
    vp = VerbPhrase()
    vp.vector = vector_from_features("verb")
    token = vector_from_features("VP")
    token.phrase = vp
    return TokenizationHypothesis(tokens=[token], confidence=1.0, description="test")


def test_policies_satisfy_protocol():
    assert isinstance(PermissiveVPPolicy(), VPGroundingPolicy)
    assert isinstance(RejectVPPolicy(), VPGroundingPolicy)


def test_default_is_host_neutral_and_custom_policy_is_scoped():
    grounder = Layer4SemanticGrounder(None)
    hypothesis = _hypothesis()
    assert grounder.ground_layer4([hypothesis]) == [hypothesis]

    with use_vp_policy(RejectVPPolicy()):
        assert grounder.ground_layer4([hypothesis]) == []

    assert grounder.ground_layer4([hypothesis]) == [hypothesis]


def test_set_active_policy_reset_restores_neutral_default():
    grounder = Layer4SemanticGrounder(None)
    hypothesis = _hypothesis()
    try:
        set_active_vp_policy(RejectVPPolicy())
        assert grounder.ground_layer4([hypothesis]) == []
    finally:
        set_active_vp_policy(None)
    assert grounder.ground_layer4([hypothesis]) == [hypothesis]
