"""The core test substrate is host-neutral and exercises every extension seam."""

from latn.lexer.scene_adapter import SceneAdapter
from latn.lexer.latn_layer_executor import tokenize_best

from tests.latn.support import TEST_DIMENSIONS, TestEntity, TestSceneAdapter


def test_neutral_fixture_activates_artificial_dimensions_and_vocabulary(neutral_latn):
    adjective = neutral_latn["bright"]
    verb = neutral_latn["inspect"]

    assert all(name.startswith("test_") for name in TEST_DIMENSIONS)
    assert adjective.isa("adj") and adjective.isa("test_quality")
    assert verb.isa("verb") and verb.isa("test_intent")


def test_neutral_adapter_satisfies_the_core_protocol(neutral_latn):
    adapter = TestSceneAdapter([TestEntity("object")])
    assert isinstance(adapter, SceneAdapter)


def test_core_literal_decoder_preserves_values_without_host_coordinates(neutral_latn):
    token = tokenize_best("[1,2,3]")[0]
    assert token.isa("literal")
    assert not token.isa("vector")
    assert token.data["values"] == (1.0, 2.0, 3.0)
