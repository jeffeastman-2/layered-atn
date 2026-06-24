"""Contract test: Layer-4 delegates the verb-phrase verdict to the *active*
VPGroundingPolicy, and the Engraf default is fail-closed.

This is the non-deferrable half of seam #3: without an injectable policy,
StrictVPPolicy's final `return False` rejects every verb phrase that
activates no CAD intent dimension -- i.e. every Driftmoor verb phrase.
Behavior-preservation of the Engraf rules themselves is covered by the
existing 265 latn tests (ground_layer4 runs them through StrictVPPolicy).
"""

from latn.lexer.hypothesis import TokenizationHypothesis
from latn.lexer.semantic_grounding_layer4 import Layer4SemanticGrounder
from latn.lexer.vector_space import vector_from_features
from latn.lexer.vp_policy import (
    StrictVPPolicy,
    PermissiveVPPolicy,
    VPGroundingPolicy,
    set_active_vp_policy,
    use_vp_policy,
)
from latn.pos.verb_phrase import VerbPhrase


def _non_cad_vp():
    """A verb phrase that activates no CAD intent dim and has no NP --
    the shape of a Driftmoor VP as far as L4 is concerned."""
    vp = VerbPhrase()
    vp.vector = vector_from_features("verb action")  # not create/style/move/tobe/...
    return vp


def _vp_hypothesis(vp):
    tok = vector_from_features("VP")
    tok.phrase = vp
    return TokenizationHypothesis(tokens=[tok], confidence=1.0, description="t")


def test_policies_satisfy_protocol():
    assert isinstance(StrictVPPolicy(), VPGroundingPolicy)
    assert isinstance(PermissiveVPPolicy(), VPGroundingPolicy)


def test_engraf_default_is_fail_closed_permissive_is_not():
    # isa() yields numpy.bool_, so assert truthiness, not `is False`.
    vp = _non_cad_vp()
    assert not StrictVPPolicy().validate_vp(vp)
    assert PermissiveVPPolicy().validate_vp(vp)


def test_injection_flows_through_the_grounder_and_is_scoped():
    grounder = Layer4SemanticGrounder(None)
    hyp = _vp_hypothesis(_non_cad_vp())

    # Default (Engraf, fail-closed): the non-CAD VP hypothesis is dropped.
    assert grounder.ground_layer4([hyp]) == []

    # Injected permissive policy: the same hypothesis survives L4.
    with use_vp_policy(PermissiveVPPolicy()):
        assert grounder.ground_layer4([hyp]) == [hyp]

    # Restored: no leak back to the default.
    assert grounder.ground_layer4([hyp]) == []


def test_set_active_vp_policy_reset():
    grounder = Layer4SemanticGrounder(None)
    hyp = _vp_hypothesis(_non_cad_vp())
    try:
        set_active_vp_policy(PermissiveVPPolicy())
        assert grounder.ground_layer4([hyp]) == [hyp]
    finally:
        set_active_vp_policy(None)  # reset to Engraf default
    assert grounder.ground_layer4([hyp]) == []
