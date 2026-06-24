"""Contract test for runtime dimension registration.

A domain (e.g. Driftmoor) must be able to extend the by-design vector space
with its own dimensions WITHOUT editing Engraf's vocabulary/schema source.
register_dimensions is the generic, domain-neutral hook for that: append
dims at startup, vectors pick up the new width live, and the new dims
participate in semantic similarity.

Each test restores global schema state so the rest of tests/latn is
unaffected (the schema is a process global).
"""

import numpy as np

from latn.An_N_Space_Model import vector_dimensions as vd
from latn.lexer import vector_space as vs
from latn.lexer.vector_space import VectorSpace, vector_from_features


def _snapshot():
    return (
        list(vd.VECTOR_DIMENSIONS),
        set(vd.SEMANTIC_DIMENSIONS),
        set(vd.POS_DIMENSIONS),
        vs.VECTOR_LENGTH,
    )


def _restore(snap):
    dims, sem, pos, length = snap
    vd.VECTOR_DIMENSIONS[:] = dims
    vd.SEMANTIC_DIMENSIONS.clear(); vd.SEMANTIC_DIMENSIONS.update(sem)
    vd.POS_DIMENSIONS.clear(); vd.POS_DIMENSIONS.update(pos)
    vs.VECTOR_LENGTH = length


def test_register_grows_schema_and_vector_width():
    snap = _snapshot()
    try:
        before = len(vd.VECTOR_DIMENSIONS)
        total = vd.register_dimensions(["rpg_test_alpha", "rpg_test_beta"])
        assert total == before + 2
        # New vectors are constructed at the new width, live.
        assert len(VectorSpace().vector) == before + 2
        # Back-compat constant kept in sync.
        assert vs.VECTOR_LENGTH == before + 2
    finally:
        _restore(snap)


def test_registered_dim_is_addressable_and_semantic():
    snap = _snapshot()
    try:
        vd.register_dimensions(["rpg_test_magical"])
        v = vector_from_features("adj", rpg_test_magical=1.0)
        assert v["rpg_test_magical"] == 1.0
        assert v.isa("rpg_test_magical")
        # Joined SEMANTIC_DIMENSIONS, so similarity uses it.
        assert "rpg_test_magical" in vd.SEMANTIC_DIMENSIONS
        assert vd.get_semantic_mask()[vd.VECTOR_DIMENSIONS.index("rpg_test_magical")]
    finally:
        _restore(snap)


def test_register_is_idempotent_and_preserves_indices():
    snap = _snapshot()
    try:
        prefix_before = list(vd.VECTOR_DIMENSIONS)
        vd.register_dimensions(["rpg_test_gamma"])
        vd.register_dimensions(["rpg_test_gamma"])  # no double-append
        assert vd.VECTOR_DIMENSIONS.count("rpg_test_gamma") == 1
        # Append-only: the existing prefix is untouched (so prior vectors keep
        # their meaning) and the new dim sits at the end.
        assert vd.VECTOR_DIMENSIONS[: len(prefix_before)] == prefix_before
        assert vd.VECTOR_DIMENSIONS[-1] == "rpg_test_gamma"
    finally:
        _restore(snap)


def test_similarity_matches_on_registered_property_dim():
    """The Driftmoor use case in miniature: a query dim grounds a candidate."""
    snap = _snapshot()
    try:
        vd.register_dimensions(["rpg_test_cleric"])
        query = vector_from_features("noun", rpg_test_cleric=1.0)   # "the cleric"
        cleric = vector_from_features("noun", rpg_test_cleric=1.0)  # a cleric entity
        fighter = vector_from_features("noun")                      # no class dim
        assert query.semantic_similarity(cleric) > query.semantic_similarity(fighter)
    finally:
        _restore(snap)
