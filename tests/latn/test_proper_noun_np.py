"""The NP ATN accepts a proper noun as an NP head.

Engraf's CAD vocabulary has no proper-noun tokens (names were only ever
created via the naming verb with quoted tokens), so this capability is for
downstream domains (e.g. Driftmoor) that refer to entities by name. A proper
noun should parse as a bare NP head and as a head after a determiner/adjective.
"""

from latn.atn.np import run_np
from latn.lexer.vector_space import vector_from_features


def _proper(word):
    tok = vector_from_features("proper_noun")
    tok.word = word
    return tok


def _adj(word):
    tok = vector_from_features("adj")
    tok.word = word
    return tok


def _det(word="the"):
    tok = vector_from_features("det def")
    tok.word = word
    return tok


def test_bare_proper_noun_forms_np():
    np = run_np([_proper("amara")])
    assert np is not None
    assert np.noun == "amara"
    assert np.vector.isa("proper_noun")


def test_proper_noun_after_determiner():
    np = run_np([_det("the"), _proper("amara")])
    assert np is not None
    assert np.noun == "amara"


def test_proper_noun_as_head_after_modifier_adjective():
    # "tall Amara" — an adjective modifier then a proper-noun head. This is the
    # path a Driftmoor dual noun/adj descriptor ("the dwarf Amara") takes in
    # modifier position. (Pure noun -> proper noun is not an NP, by design.)
    np = run_np([_adj("tall"), _proper("amara")])
    assert np is not None
    assert np.noun == "amara"
