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
from tests.latn.support import TestEntity, TestSceneAdapter


def test_stub_satisfies_protocol_structurally(neutral_latn):
    assert isinstance(TestSceneAdapter([]), SceneAdapter)
    assert isinstance(TestEntity("object", object_id="object_1"), GroundedEntity)


def test_executor_grounds_descriptive_np_through_the_protocol(neutral_latn):
    entity = TestEntity("object", object_id="object_1")
    stub = TestSceneAdapter([entity])
    executor = LATNLayerExecutor(stub)

    result = executor.execute_layer2("the object")

    assert result.success
    grounded = [
        e
        for gr in result.grounding_results
        if gr.success
        for e in (gr.resolved_objects or [])
    ]
    assert entity in grounded, "NP was not grounded via the SceneAdapter protocol"


def test_executor_runs_without_an_adapter(neutral_latn):
    """No adapter -> parse-only, no grounding, no crash (the optional seam)."""
    result = LATNLayerExecutor().execute_layer2("the object")
    assert result.success
