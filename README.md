# layered-atn
LATN (Layered ATN) parses natural language through staged augmented transition networks — tokenize → noun / prepositional / verb phrases → sentence — scoring competing hypotheses in a vector-space model. Its built-in dimensions and vocabulary are grammatical only. A host supplies content vocabulary, executable semantic dimensions, world grounding, and any validation policies. The result is grounded, structured intent, not just a parse tree.

## The layers

| Layer | Builds | Result |
|-------|--------|--------|
| L1 | tokens (lexical) | `Layer1Result` |
| L2 | noun phrases (NP) | `Layer2Result` |
| L3 | prepositional phrases (PP) | `Layer3Result` |
| L4 | verb phrases (VP) | `Layer4Result` |
| L5 | sentence phrases (SP) | `Layer5Result` |

Each layer tokenizes and grounds on top of the one below; higher layers fold the
lower phrases into composite tokens, carrying a semantic vector through the whole
stack and keeping multiple hypotheses alive until grounding resolves them.

## The seams

A host plugs into LATN through two **substrate** seams and a family of
**per-layer grounding** seams — none of which require LATN to know anything
domain-specific.

### Substrate

| Seam | What the host supplies | Entry point |
|------|------------------------|-------------|
| Lexicon | its vocabulary (words → feature vectors), activated per parse | `latn.lexer.lexicon` — `Lexicon`, `use_lexicon` |
| Dimension schema | executable host axes appended to the grammatical schema | `latn.An_N_Space_Model.vector_dimensions` — `register_dimensions` |

### Grounding — one seam per grounding layer

| Layer | Grounding seam | Entry point |
|-------|----------------|-------------|
| L2 — noun phrases | **`SceneAdapter`** (the host's world bridge) | `latn.lexer.scene_adapter` — `SceneAdapter`, `GroundedEntity` |
| L3 — PP relationships | `SpatialPolicy` | `PermissiveSpatialPolicy`, `use_spatial_policy` |
| L4 — verb phrases | `VPGroundingPolicy` | `PermissiveVPPolicy`, `use_vp_policy` |
| L5 — sentence phrases | `SPGroundingPolicy` | `PermissiveSPPolicy`, `use_sp_policy` |

Two shapes within the family. The **NP seam is a resolver**: the `SceneAdapter`
the host implements *is* the bridge to its entities (`resolve_noun_phrase`,
`resolve_pronoun`), and it is required. The **L3–L5 seams are filters**: thin,
swappable policies that toggle **fail-closed** (reject ungrounded phrases) vs
**permissive**, with host-neutral permissive defaults. A host mixes them to taste — e.g.
Driftmoor runs permissive VP + SP so it can extract a verb phrase even when full
sentence grounding would reject it, then grounds noun phrases through its own
`SceneAdapter`.

## Install

```bash
pip install -e .      # editable, for development
```

Runtime is numpy-only. (The CAD/visualizer and LLM extras the code originally
shipped with stayed behind in Engraf — see Provenance.)

## Quick start

```python
from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.lexer.scene_adapter import SceneAdapter

class MyScene(SceneAdapter):
    ...  # ground phrases to your domain's entities

ex = LATNLayerExecutor(scene_model=MyScene())
result = ex.execute_layer5(sentence)
# result.sentence_phrases -> grounded sentence(s); each SP carries its predicate (the VP)
```

Wiring in the other two seams:

```python
from latn.An_N_Space_Model.vector_dimensions import register_dimensions
from latn.lexer.lexicon import use_lexicon

register_dimensions(...)        # add executable domain axes
with use_lexicon(my_lexicon):   # activate your vocabulary for the parse
    result = ex.execute_layer5(sentence)
```

A complete integration — lexicon builder, scene adapter, dimension registration,
and parse → application intent — lives in the **Driftmoor** project's
`engine/latn*.py` (a single `engine/latn.py` re-exports LATN's whole surface).

## Vector space

The base space contains only named grammatical distinctions used by tokenization,
agreement, morphology, and ATN transitions. Its semantic mask is empty. Hosts
append their executable dimensions at startup; those dimensions alone participate
in semantic similarity. This is the schema-level application of Achem's Razor.

## Provenance

Extracted from the **Engraf** natural-language CAD project, where it grew as the
parsing core. It is now standalone so any host can use it: Engraf consumes it for
scene construction, and **Driftmoor** (a multi-agent D&D engine) uses it as the
natural-language front end that turns player and NPC English into action intents.
