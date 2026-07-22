"""Host policy seam for Layer-5 sentence-phrase acceptance."""

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
        _active = PermissiveSPPolicy()
    return _active


def set_active_sp_policy(policy: Optional[SPGroundingPolicy]) -> None:
    """Swap the active policy. Pass None to reset to the neutral default."""
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
    "PermissiveSPPolicy",
    "get_active_sp_policy",
    "set_active_sp_policy",
    "use_sp_policy",
]
