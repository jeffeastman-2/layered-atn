from latn.lexer.grounding_promise import GroundingPromise
from latn.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder
from latn.pos.noun_phrase import NounPhrase
from latn.lexer.vector_space import VectorSpace


class Entity:
    def __init__(self, name):
        self.name = self.object_id = self.entity_id = name
        self.vector = VectorSpace()
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}


class ChangingAdapter:
    def __init__(self):
        self.grounding_revision = 0
        self.entities = []
        self.recent = []

    def resolve_noun_phrase(self, np):
        return [(1.0, obj) for obj in self.entities if obj.name == np.noun]

    def resolve_pronoun(self, pronoun):
        return self.recent[-1:] if pronoun == "it" else []


def test_unresolved_np_promise_resolves_after_world_revision():
    adapter = ChangingAdapter()
    np = NounPhrase("box")
    promise = Layer2SemanticGrounder(adapter)._promise(np)

    assert promise.get("scene_objects") == []
    box = Entity("box")
    adapter.entities.append(box)
    adapter.grounding_revision += 1

    assert promise.get("scene_objects") == [box]


def test_pronoun_promise_tracks_discourse_revision():
    adapter = ChangingAdapter()
    np = NounPhrase()
    np.pronoun = "it"
    np.vector["pronoun"] = 1.0
    np.vector["singular"] = 1.0
    promise = Layer2SemanticGrounder(adapter)._promise(np)

    assert promise.get("scene_objects") == []
    box = Entity("box")
    adapter.recent.append(box)
    adapter.grounding_revision += 1

    assert promise.get("scene_objects") == [box]


def test_promise_without_revision_refreshes_at_execution_boundary():
    calls = []
    adapter = object()
    promise = GroundingPromise(adapter, lambda: {"scene_objects": [len(calls)] if not calls.append(1) else []})

    first = promise.get("scene_objects")
    cached = promise.get("scene_objects")
    second = promise.force(refresh=True).get("scene_objects")

    assert first == cached
    assert first != second
