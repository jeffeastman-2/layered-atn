"""Contract test: the LATN core grounds against the SceneAdapter *protocol*,
not against Engraf's concrete SceneModel.

Engraf's own 256 latn tests prove no regression of the one concrete adapter
(SceneModel). They do NOT prove the abstraction is real, because they only
ever exercise that single implementation. This test feeds the executor a
hand-rolled stub adapter that has nothing to do with SceneModel, proving the
seam is genuine. It is the precondition for Driftmoor supplying its own
adapter in phase 2.
"""

from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.lexer.scene_adapter import SceneAdapter, GroundedEntity
from latn.lexer.vector_space import VectorSpace


class StubEntity:
    """Minimal GroundedEntity with no relationship to SceneObject."""

    def __init__(self, name, oid):
        self.name = name
        self.vector = VectorSpace()
        self.object_id = oid
        self.entity_id = oid
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}


class StubScene:
    """Minimal SceneAdapter. Records calls; resolves by exact noun match."""

    def __init__(self, entities):
        self._entities = entities
        self.np_calls = 0
        self.pronoun_calls = 0

    def resolve_noun_phrase(self, np):
        self.np_calls += 1
        return [(1.0, e) for e in self._entities if e.name == np.noun]

    def resolve_pronoun(self, pronoun):
        self.pronoun_calls += 1
        return list(self._entities)


def test_stub_satisfies_protocol_structurally():
    assert isinstance(StubScene([]), SceneAdapter)
    assert isinstance(StubEntity("box", "box_1"), GroundedEntity)


def test_executor_grounds_descriptive_np_through_the_protocol():
    box = StubEntity("box", "box_1")
    stub = StubScene([box])
    executor = LATNLayerExecutor(stub)

    result = executor.execute_layer2("the box")

    assert result.success
    assert stub.np_calls > 0, "core did not call the adapter's resolve_noun_phrase"
    grounded = [
        e
        for gr in result.grounding_results
        if gr.success
        for e in (gr.resolved_objects or [])
    ]
    assert box in grounded, "NP was not grounded to the stub entity via the protocol"


def test_executor_runs_without_an_adapter():
    """No adapter -> parse-only, no grounding, no crash (the optional seam)."""
    result = LATNLayerExecutor().execute_layer2("the box")
    assert result.success
