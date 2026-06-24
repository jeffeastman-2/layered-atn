"""SceneAdapter: the sole boundary between the LATN core (L1-5) and a world.

The LATN parser must not depend on any concrete scene/world model. Every
scene touch made by the L1-5 grounders bottoms out in the two methods of
`SceneAdapter` below; every entity attribute they read is declared on
`GroundedEntity`. Engraf's SceneModel implements this; Driftmoor supplies
its own perception-backed implementation.

Derived from the only real scene call sites in the core:
  - semantic_grounding_layer2.py:65  resolve_pronoun(...)
  - semantic_grounding_layer2.py:139 find_noun_phrase(np, return_all_matches=True)
  - spatial_validation.py:156-159    obj.position['x'|'y'|'z']  (L3 path)
"""

from typing import List, Mapping, Optional, Protocol, Tuple, runtime_checkable

from latn.lexer.vector_space import VectorSpace
from latn.pos.noun_phrase import NounPhrase


@runtime_checkable
class GroundedEntity(Protocol):
    """A world thing the parser can resolve a NounPhrase/pronoun to.

    Closed contract: every member is read by the L1-5 core. `name`/`vector`
    are consumed by the adapter's own matching; `object_id`/`entity_id` are
    read back by Layer-2 grounding; `position` is read by the Layer-3 spatial
    validator (only when spatial PPs are present). `entity_id`/`object_id`
    duplication mirrors an existing Engraf wart and is preserved here to keep
    this step behavior-preserving; unify later in a non-behavioral pass.
    """

    name: str
    vector: VectorSpace
    object_id: str
    entity_id: str
    position: Mapping[str, float]  # keys: 'x', 'y', 'z'


@runtime_checkable
class SceneAdapter(Protocol):
    """Everything L1-5 needs from a world. Read-only: the core never mutates
    the world (L5 grounding only filters well-formed hypotheses; execution
    lives in engraf/interpreter, outside the extracted core).
    """

    def resolve_noun_phrase(
        self, np: NounPhrase
    ) -> List[Tuple[float, GroundedEntity]]:
        """Rank world entities matching a descriptive NP, best first.

        Replaces SceneModel.find_noun_phrase(np, return_all_matches=True).
        The matching algorithm is the adapter's concern; grammatical-number
        selection (singular -> first, plural -> all) stays in Layer 2.
        """
        ...

    def resolve_pronoun(self, pronoun: str) -> List[GroundedEntity]:
        """Resolve 'it'/'they'/'them' to entities (unranked).

        Replaces the module function engraf.visualizer.scene.scene_model
        .resolve_pronoun. The recency/perception model is the adapter's
        concern. May raise ValueError on an unrecognized pronoun (Layer 2
        already catches this).
        """
        ...


__all__ = ["GroundedEntity", "SceneAdapter"]
