# Achem's Razor

**A guiding design principle for the Layered Augmented Transition Network (LATN)**

---

## Abstract

As LATN evolved beyond its origins in the 1975 ENGRAF system, a recurring architectural question emerged:

> *When should a new semantic vector dimension be introduced?*

The answer became the central design rule of LATN.

This rule, informally named **Achem's Razor**, governs not only the growth of the vector space but the separation of responsibility throughout the architecture.

---

# Achem's Razor

> **Introduce no semantic vector dimension unless changing that dimension changes the computation.**

Or equivalently,

> **A semantic distinction should exist only if some executable behavior depends upon it.**

Unlike traditional linguistic feature systems, LATN vectors are not intended to be complete semantic descriptions of words.

They are executable representations.

Every dimension exists because some downstream computation depends upon it.

If changing a dimension never changes the behavior of either the parser or the semantic engine, that dimension should not exist.

---

# Design Philosophy

Traditional semantic systems often ask:

> *What properties does this word have?*

LATN asks a different question:

> *What distinctions must exist for the computation to behave differently?*

This shifts the design from descriptive semantics to executable semantics.

Vector dimensions are introduced only when they distinguish behavior.

---

# Three Levels of Meaning

Achem's Razor naturally divides the architecture into three distinct layers.

## 1. LATN Core

LATN owns only those distinctions required for language interpretation.

Examples include:

- part of speech
- plurality
- agreement
- pronoun classes
- determiner behavior
- grammatical distinctions required by ATN transitions

These dimensions exist because parser behavior depends upon them.

LATN deliberately avoids host-specific semantic information.

---

## 2. Host Adapters

Applications extend LATN with dimensions required by their own semantic engines.

For example:

### ENGRAF

Additional dimensions might distinguish:

- translation
- rotation
- creation
- deletion

These exist because the graphics engine performs different operations.

### Driftmoor

Additional dimensions might distinguish:

- locomotion
- purchasing
- transferring ownership
- speech acts

These exist because the simulation behaves differently for each action.

LATN itself never interprets these dimensions.

It merely transports vectors through the parser.

Grounding belongs entirely to the host.

---

## 3. Persistent World Models

Many properties should not exist in lexical vectors at all.

Examples include:

- RGB values
- inventories
- ownership
- world coordinates
- memories
- relationships
- economic values
- health
- reputation

These are properties of persistent entities.

They belong in the world model, not in the language.

---

# Behavioral Vectors

LATN vectors should be viewed as behavioral descriptions rather than property descriptions.

A vector dimension answers the question:

> *Would the computation behave differently if this value changed?*

If the answer is "no," the dimension probably should not exist.

This makes LATN vectors fundamentally different from:

- symbolic feature systems
- ontology databases
- property dictionaries

The vector space represents computational distinctions rather than descriptive attributes.

---

# Sparse by Design

LATN vectors often appear "Boolean-like."

Many dimensions contain values such as:

```
0
1
0
0
1
0
```

This is intentional.

The goal is not compression.

The goal is interpretability.

Each dimension has an explicit computational purpose.

Modern LLM embedding spaces contain thousands of dimensions whose meanings are largely opaque.

LATN takes the opposite approach.

Its vector dimensions are sparse, interpretable, and introduced only when required by computation.

---

# Core Vocabulary vs Host Vocabulary

LATN contains only vocabulary necessary for language interpretation.

Host applications contribute vocabulary required for semantic grounding.

The same English word may therefore have different meanings in different hosts.

Example:

```
move
```

In ENGRAF:

```
move
→ geometric translation
```

In Driftmoor:

```
move
→ locomotion
```

LATN owns neither interpretation.

It merely recognizes the grammatical behavior of the word.

---

# Separation of Responsibility

The complete architecture becomes:

```
Natural Language
        │
        ▼
      LATN
(Intent Interpretation)
        │
        ▼
   Typed Intent
        │
        ▼
 Host Semantic Engine
        │
        ▼
 Persistent World
        │
        ▼
 Narrative
```

Each layer has a single responsibility.

LATN determines intent.

The semantic engine determines meaning.

The persistent world determines reality.

The language model explains the results.

---

# Consequences

Achem's Razor has several important consequences.

## The vector space grows slowly.

New dimensions require explicit computational justification.

---

## Host applications remain independent.

LATN does not accumulate application-specific semantics.

---

## Persistent world models remain authoritative.

Reality belongs to the simulation, not the parser.

---

## New applications become easy to build.

A new host contributes only the semantic dimensions it actually needs.

---

## The architecture remains understandable.

Every vector dimension has a documented behavioral purpose.

No dimension exists "just in case."

---

# Relationship to ENGRAF

ENGRAF demonstrated that executable semantic vectors could drive a graphics system.

LATN generalizes that idea.

Rather than encoding graphics semantics directly, LATN provides a host-independent language interpretation engine whose vectors can be extended by arbitrary semantic domains.

This preserves the original insight while allowing applications far beyond computer graphics.

---

# Ellipses, Not Epicycles

Architectural evolution should simplify the system rather than accumulate special cases.

When a new abstraction removes multiple exceptions, it is probably the correct abstraction.

When solving a problem requires adding exceptions instead of eliminating them, the architecture should be reconsidered.

This principle motivated the separation of:

- ENGRAF
- LATN
- Driftmoor
- WorldBuilder

into distinct systems with clearly defined responsibilities.

---

# Summary

Achem's Razor is more than a rule for adding vector dimensions.

It is the architectural philosophy of LATN.

Each layer introduces only the distinctions required for its own computation.

The result is a language interpretation system that remains small, deterministic, interpretable, and extensible while supporting increasingly sophisticated semantic hosts.

The simplest architecture is not the one with the fewest features.

It is the one in which every feature earns its existence.
