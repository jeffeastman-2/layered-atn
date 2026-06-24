# layered-atn
LATN (Layered ATN) parses natural language through staged augmented transition networks — tokenize → noun / prepositional / verb phrases → sentence — scoring competing hypotheses in a vector-space semantic model. It's host-agnostic by design: the host configures it through two substrate seams — its own lexicon (vocabulary) and extra axes on the base 70-dimensional vector-space schema (the semantic dimensions the model reasons over) — plus a grounding seam at each layer: a SceneAdapter resolves noun phrases to entities in the host's world, while strict-or-permissive policies govern how prepositional, verb, and sentence phrases bind. The result is grounded, structured intent, not just a parse tree.

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
| Dimension schema | extra semantic axes on top of the 70 base dims | `latn.An_N_Space_Model.vector_dimensions` — `register_dimensions` |

### Grounding — one seam per grounding layer

| Layer | Grounding seam | Entry point |
|-------|----------------|-------------|
| L2 — noun phrases | **`SceneAdapter`** (the host's world bridge) | `latn.lexer.scene_adapter` — `SceneAdapter`, `GroundedEntity` |
| L3 — PP / spatial | `SpatialPolicy` | `StrictSpatialPolicy`, `PermissiveSpatialPolicy`, `use_spatial_policy` |
| L4 — verb phrases | `VPGroundingPolicy` | `StrictVPPolicy`, `PermissiveVPPolicy`, `use_vp_policy` |
| L5 — sentence phrases | `SPGroundingPolicy` | `StrictSPPolicy`, `PermissiveSPPolicy`, `use_sp_policy` |

Two shapes within the family. The **NP seam is a resolver**: the `SceneAdapter`
the host implements *is* the bridge to its entities (`resolve_noun_phrase`,
`resolve_pronoun`), and it is required. The **L3–L5 seams are filters**: thin,
swappable policies that toggle **fail-closed** (reject ungrounded phrases) vs
**permissive**, each with a strict default. A host mixes them to taste — e.g.
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
result = ex.execute_layer5("draw a red cube above the table")
# result.sentence_phrases -> grounded sentence(s); each SP carries its predicate (the VP)
```

Wiring in the other two seams:

```python
from latn.An_N_Space_Model.vector_dimensions import register_dimensions
from latn.lexer.lexicon import use_lexicon

register_dimensions(...)        # add domain axes on top of the 70-dim base space
with use_lexicon(my_lexicon):   # activate your vocabulary for the parse
    result = ex.execute_layer5(sentence)
```

A complete integration — lexicon builder, scene adapter, dimension registration,
and parse → application intent — lives in the **Driftmoor** project's
`engine/latn*.py` (a single `engine/latn.py` re-exports LATN's whole surface).

## Vector space

The base semantic space is **70 dimensions** — POS markers, grammatical features,
verb-intent vectors, 3D spatial coordinates, semantic-preposition axes, and more
— of which ~65 carry semantic weight (punctuation slots are excluded from
similarity). Hosts append their own dimensions at startup; the space is open by
design.

## Provenance

Extracted from the **Engraf** natural-language CAD project, where it grew as the
parsing core. It is now standalone so any host can use it: Engraf consumes it for
scene construction, and **Driftmoor** (a multi-agent D&D engine) uses it as the
natural-language front end that turns player and NPC English into action intents.
