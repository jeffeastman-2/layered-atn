"""Host-neutral fixtures for exercising the LATN core.

Nothing in this module is production vocabulary or a proposed semantic
ontology.  The deliberately ``test_*`` dimensions prove that LATN transports
host extensions without making their meanings part of the parser.
"""

from dataclasses import dataclass, field

from latn.An_N_Space_Model.vector_dimensions import register_dimensions
from latn.lexer.lexicon import Lexicon
from latn.lexer.vector_space import VectorSpace, vector_from_features


TEST_DIMENSIONS = (
    "test_quality",
    "test_relation",
    "test_intent",
)


def build_test_lexicon() -> Lexicon:
    """Return a fresh, minimal English lexicon with artificial semantics."""
    register_dimensions(TEST_DIMENSIONS)

    vf = vector_from_features
    table = {
        # Closed-class grammar used throughout the layer tests.
        "a": vf("det singular", number=1.0),
        "an": vf("det singular", number=1.0),
        "the": vf("det def"),
        "one": vf("det def singular", number=1.0),
        "two": vf("det def plural", number=2.0),
        "and": vf("conj and"),
        "or": vf("conj or"),
        "not": vf("neg"),
        "no": vf("neg"),
        "is": vf("tobe"),
        "are": vf("tobe"),
        "was": vf("tobe"),
        "with": vf("prep"),
        "from": vf("prep"),
        "to": vf("prep"),
        "on": vf("prep", test_relation=1.0),
        "above": vf("prep", test_relation=1.0),
        "below": vf("prep", test_relation=-1.0),
        "behind": vf("prep", test_relation=-1.0),
        "near": vf("prep", test_relation=1.0),
        "than": vf("prep"),
        "very": vf("adv", adverb=1.5),
        "extremely": vf("adv", adverb=2.0),
        "slightly": vf("adv", adverb=0.75),
        ",": vf("punct comma"),
        ".": vf("punct period"),
        "?": vf("punct question"),
        "!": vf("punct exclaim"),

        # Content words have only grammar plus visibly artificial test axes.
        "object": vf("noun singular"),
        "item": vf("noun singular"),
        "person": vf("noun singular"),
        "table": vf("noun singular"),
        "chair": vf("noun singular"),
        "sphere": vf("noun singular"),
        "box": vf("noun singular"),
        "cylinder": vf("noun singular"),
        "cube": vf("noun singular"),
        "house": vf("noun singular"),
        "circle": vf("noun singular"),
        "octahedron": vf("noun singular"),
        "light house": vf("noun singular"),
        "red": vf("adj", test_quality=1.0),
        "blue": vf("adj", test_quality=1.0),
        "green": vf("adj", test_quality=1.0),
        "big": vf("adj", test_quality=1.0),
        "large": vf("adj", test_quality=1.0),
        "small": vf("adj", test_quality=-1.0),
        "tall": vf("adj", test_quality=1.0),
        "bright": vf("adj", test_quality=1.0),
        "rough": vf("adj", test_quality=1.0),
        "smooth": vf("adj", test_quality=1.0),
        "inspect": vf("verb", test_intent=1.0),
        "modify": vf("verb", test_intent=1.0),
        "call": vf("verb", test_intent=1.0),
        "name": vf("verb", test_intent=1.0),
        "it": vf("pronoun singular"),
        "they": vf("pronoun plural"),
        "them": vf("pronoun plural"),
    }
    return Lexicon(table)


class TestVPPolicy:
    """Accept structurally recognizable verb phrases, independent of intent."""

    __test__ = False

    def validate_vp(self, vp) -> bool:
        return bool(vp and (getattr(vp, "verb", None) or vp.vector.isa("tobe")))


class TestSPPolicy:
    """Accept a hypothesis whenever the grammar produced a sentence phrase."""

    __test__ = False

    def accept_hypothesis(self, sentence_phrases, all_tokens_are_sp) -> bool:
        return bool(sentence_phrases)


class TestSpatialPolicy:
    """A deterministic policy used only to prove policy delegation."""

    __test__ = False

    def applies_to(self, pp_token) -> bool:
        return pp_token.isa("test_relation")

    def validate(self, pp_token, obj1, obj2) -> bool:
        return bool(pp_token.isa("test_relation"))


@dataclass
class TestEntity:
    """Small GroundedEntity-shaped value for adapter contract tests."""

    __test__ = False

    name: str
    vector: VectorSpace = field(default_factory=VectorSpace)
    object_id: str = ""
    entity_id: str = ""
    position: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})

    def __post_init__(self):
        self.object_id = self.object_id or self.name
        self.entity_id = self.entity_id or self.object_id


class TestSceneAdapter:
    """Name-based neutral adapter; no CAD scene behavior is implied."""

    __test__ = False

    def __init__(self, entities=()):
        self.entities = list(entities)
        self.recent = list(self.entities)

    def resolve_noun_phrase(self, np):
        noun = (getattr(np, "noun", "") or "").lower()
        return [
            (1.0, entity)
            for entity in self.entities
            if entity.name.lower() == noun
        ]

    def resolve_pronoun(self, pronoun):
        word = (pronoun or "").lower()
        if word == "it":
            return self.recent[-1:] if self.recent else []
        if word in {"they", "them"}:
            return list(self.entities)
        return []
