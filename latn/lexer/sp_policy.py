"""Injectable Layer-5 sentence-phrase grounding policy.

The Layer-5 analog of vp_policy. Layer-5's ``ground_layer5`` decides which
tokenization hypotheses survive as "sentences". Engraf's rule is FAIL-CLOSED
at the hypothesis level: a hypothesis grounds only if EVERY token collapsed
into a sentence phrase (or a conjunction of them) -- any leftover non-SP token
rejects the whole hypothesis. That is right for Engraf's complete-utterance
CAD grammar, but it discards partial parses a domain consumer may still want:
e.g. an imperative whose trailing material didn't fully fold into the SP, or
verbose prose with one well-formed clause.

EngrafSPPolicy below preserves that strict behavior verbatim (and is the
default, so Engraf is unaffected). A consumer (Driftmoor) can activate
PermissiveSPPolicy so a hypothesis survives whenever it produced at least one
sentence phrase, even with leftover tokens -- the SP-level counterpart to
PermissiveVPPolicy. The consumer then uses the SP structure where it formed
and can fall back to the lower-layer VP elsewhere, instead of losing the
parse outright.
"""

from typing import List, Optional, Protocol, runtime_checkable

from latn.pos.sentence_phrase import SentencePhrase


@runtime_checkable
class SPGroundingPolicy(Protocol):
    """Decides whether a Layer-5 hypothesis is well-formed for the active
    domain. The grounder owns hypothesis iteration and SP extraction; this owns
    the keep/reject verdict given what was extracted."""

    def accept_hypothesis(self, sentence_phrases: List[SentencePhrase],
                          all_tokens_are_sp: bool) -> bool:
        ...


class EngrafSPPolicy:
    """Engraf's complete-sentence rule: keep a hypothesis only if it produced
    at least one sentence phrase AND every token folded into one (no leftover
    non-SP tokens). Behavior-preserving for the previous hardcoded
    ground_layer5 logic."""

    def accept_hypothesis(self, sentence_phrases: List[SentencePhrase],
                          all_tokens_are_sp: bool) -> bool:
        return bool(sentence_phrases) and all_tokens_are_sp


class PermissiveSPPolicy:
    """Keep any hypothesis that produced at least one sentence phrase, even if
    non-SP tokens remain. The SP analog of PermissiveVPPolicy, for domains
    whose partial-parse handling is adjudicated downstream."""

    def accept_hypothesis(self, sentence_phrases: List[SentencePhrase],
                          all_tokens_are_sp: bool) -> bool:
        return bool(sentence_phrases)


_active: Optional[SPGroundingPolicy] = None


def get_active_sp_policy() -> SPGroundingPolicy:
    """The policy L5 consults. Call at use-time (not import-bound) so
    set_active_sp_policy / use_sp_policy take effect."""
    global _active
    if _active is None:
        _active = EngrafSPPolicy()
    return _active


def set_active_sp_policy(policy: Optional[SPGroundingPolicy]) -> None:
    """Swap the active policy. Pass None to reset to the Engraf default."""
    global _active
    _active = policy


class use_sp_policy:
    """Context manager: activate `policy` for a scope, restore the prior one.
    For per-parse activation (Driftmoor) and hermetic tests."""

    def __init__(self, policy: SPGroundingPolicy):
        self._new = policy
        self._prev: Optional[SPGroundingPolicy] = None

    def __enter__(self) -> SPGroundingPolicy:
        global _active
        self._prev = _active
        _active = self._new
        return self._new

    def __exit__(self, *exc) -> None:
        global _active
        _active = self._prev


__all__ = [
    "SPGroundingPolicy",
    "EngrafSPPolicy",
    "PermissiveSPPolicy",
    "get_active_sp_policy",
    "set_active_sp_policy",
    "use_sp_policy",
]
