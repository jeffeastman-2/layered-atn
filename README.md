# layered-atn
LATN (Layered ATN) parses natural language through staged augmented transition networks — tokenize → noun / prepositional / verb phrases → sentence — scoring competing hypotheses in a vector-space semantic model. It's host-agnostic by design, with three injection seams: the host supplies its own lexicon (vocabulary), extends the base 70-dimensional vector-space schema (the semantic axes the model reasons over), and provides a SceneAdapter that grounds phrases to real entities in the host's world model — with pluggable grounding policies controlling how strict that matching is. The result is grounded, structured intent, not just a parse tree.

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

## The three seams

| Seam | What the host supplies | Entry point |
|------|------------------------|-------------|
| Lexicon | its vocabulary (words → feature vectors), activated per parse | `latn.lexer.lexicon` — `Lexicon`, `use_lexicon` |
| Vector-space dimension schema | extra semantic axes on top of the 70 base dims | `latn.An_N_Space_Model.vector_dimensions` — `register_dimensions` |
| Scene adapter | grounding of phrases to real entities (+ grounding policies) | `latn.lexer.scene_adapter` — `SceneAdapter`, `GroundedEntity` |

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
