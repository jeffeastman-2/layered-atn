"""Injectable Layer-4 verb-phrase grounding policy.

Phase-1 seam 3b of factoring the LATN core out of Engraf. Layer-4's
verb-phrase validation is the one place the dimension *ontology* becomes
branching logic: EngrafVPPolicy below is keyed to the CAD verb-intent
dimensions (create/style/move/rotate/scale/organize/edit/select/naming/
tobe) and is FAIL-CLOSED -- its final `return False` rejects any VP that
activates none of those. A Driftmoor verb phrase ("shoot an arrow at the
shadowbeast") activates no CAD dim, so under the Engraf policy L4 would
reject every Driftmoor VP. Making the policy injectable is therefore the
NON-deferrable half of seam #3 (unlike #3a, the dimension schema).

EngrafVPPolicy is the previous Layer4SemanticGrounder.validate_vp /
validate_vp_with_np logic, moved here verbatim -> behavior-preserving for
Engraf. Driftmoor supplies its own policy (or PermissiveVPPolicy) so its
verb phrases survive L4 and reach the DM adjudicator.
"""

from typing import Optional, Protocol, runtime_checkable

from latn.pos.noun_phrase import NounPhrase
from latn.pos.verb_phrase import VerbPhrase


@runtime_checkable
class VPGroundingPolicy(Protocol):
    """Decides whether a verb phrase is semantically well-formed for the
    active domain. The grounder owns hypothesis iteration; this owns the
    per-VP verdict."""

    def validate_vp(self, vp: VerbPhrase) -> bool:
        ...


class EngrafVPPolicy:
    """Engraf's CAD verb-phrase rules. Verbatim move of the former
    Layer4SemanticGrounder.validate_vp / validate_vp_with_np. Fail-closed."""

    def validate_vp_with_np(self, vp: VerbPhrase, np: NounPhrase) -> bool:
        vp_has_pp  = len(vp.prepositions) > 0
        vp_has_adj = len(vp.adjective_complements) > 0
        vp_has_amount = vp.amount is not None
        # Check if NP has embedded adjective semantics (e.g., "them more transparent")
        np_has_adj_semantics = np.vector.isa("adj") if np.vector else False
        # Check if PP is a comparative "than" construction
        vp_has_comparative_pp = vp_has_pp and any(
            pp.preposition == "than" or (pp.vector and pp.vector.isa("relational_comparison"))
            for pp in vp.prepositions
        )

        if (vp_has_adj and vp_has_pp) or (vp_has_adj and vp_has_amount) or (vp_has_pp and vp_has_amount):
            # multiple vp complements not allowed - BUT allow adj in NP + comparative PP
            if not (np_has_adj_semantics and vp_has_comparative_pp):
                return False
        np_preps_have_spatial_pp = False
        for prep in np.prepositions:
            if prep.vector.isa("spatial_location") or prep.vector.isa("spatial_proximity"):
                np_preps_have_spatial_pp = True
                break

        # --- STYLE / STATE-CHANGE: make, color, texture ---
        # Expect: grounded NP + adjective complement (resulting state).
        # Also accept: grounded NP with embedded adjective + comparative PP (e.g., "make them more transparent than X")
        # NOTE: color/texture carry both "transform" and "style"; prefer style rule.
        if vp.vector.isa("style") or (vp.vector.isa("transform") and not (
            vp.vector.isa("move") or vp.vector.isa("rotate") or vp.vector.isa("scale")
        )):
            if np.grounding and vp_has_adj:  # must operate on an existing object and needs adjective complement
                return True
            # Accept comparative construction: NP with adj semantics + "than" PP
            if np.grounding and np_has_adj_semantics and vp_has_comparative_pp:
                return True

        # TODO: refactor the below so that verbs may have multiple senses (e.g., "make" can be transform, style or create)
        # and we can try multiple interpretations e.g. "make the box red" -> style, "make a box on the table" -> create

        # --- MOTION / ORIENTATION / SIZE: move, rotate, scale (incl. x/y/zrotate) ---
        # Expect: grounded NP + PP (to/by/around/etc.). Adjective complements not appropriate here.
        if vp.vector.isa("move") or vp.vector.isa("rotate") or vp.vector.isa("scale"):
            if not np.grounding:
                return False
            if not vp_has_pp:
                return False
            if vp_has_adj:
                return False
            return True

        # --- ORGANIZE: align, position, group, ungroup ---
        # Expect: grounded NP; PP often present (align with, position at, group into) but not required.
        if vp.vector.isa("organize"):
            if not np.grounding:
                return False
            return True

        # --- EDIT / SELECT / NAMING: delete/copy/remove/paste, select, call/name ---
        # Expect: grounded NP; PP optional (e.g., remove from, paste into). Adjective complement not used.
        if vp.vector.isa("edit") or vp.vector.isa("select") or vp.vector.isa("naming"):
            if not np.grounding:
                return False
            return True

        # --- CREATE: create, draw, build, place (introduce new object) ---
        # Expect: UNgrounded NP (type introduction). Adj complement disallowed here;
        # PP (on/in/above) is allowed but not required.
        if vp.vector.isa("create"):
            if np.grounding:
                is_ok = False
                if vp_has_pp:
                    if vp.prepositions[0].vector.isa("spatial_location") or vp.prepositions[0].vector.isa("spatial_proximity"):
                        # grounded NP with spatial PP - probably a location specifier, not object type
                        # unground the NP and allow further processing
                        np.grounding = None
                        is_ok = True
                if not is_ok:
                    return False
            if vp_has_adj:
                return False
            # PP optional for placement; both "draw a cube" and "draw a cube on the table" are ok.
            # if not vp_has_pp and np_preps_have_spatial_pp the PP was
            # probably intended to specify location, so reject if missing.
            if not vp_has_pp and np_preps_have_spatial_pp:
                return False
            return True
        if vp.vector.isa("tobe"):
            # "is" can be used with or without NP
            if vp_has_adj and np.grounding:
                return True
            if vp_has_pp and np.grounding:
                return True
            if not vp_has_adj and not vp_has_pp and np.grounding:
                # simple "is" with grounded NP, e.g. "the box is"
                return True
            return False
        # Fallback: fail if no specific constraints apply.
        return False

    def validate_vp(self, vp: VerbPhrase) -> bool:
        np = vp.noun_phrase
        if np:
            ok = np.evaluate_boolean_function(lambda np: self.validate_vp_with_np(vp, np))
            return ok
        return vp.vector.isa("tobe")  # allow "tobe" without NP (e.g. "is above the table")


class PermissiveVPPolicy:
    """Accept every verb phrase. For domains (e.g. Driftmoor) whose verb
    semantics are adjudicated downstream rather than by L4's CAD rules;
    also the seam's test fixture."""

    def validate_vp(self, vp: VerbPhrase) -> bool:
        return True


_active: Optional[VPGroundingPolicy] = None


def get_active_vp_policy() -> VPGroundingPolicy:
    """The policy L4 consults. Call at use-time (not import-bound) so
    set_active_vp_policy / use_vp_policy take effect."""
    global _active
    if _active is None:
        _active = EngrafVPPolicy()
    return _active


def set_active_vp_policy(policy: Optional[VPGroundingPolicy]) -> None:
    """Swap the active policy. Pass None to reset to the Engraf default."""
    global _active
    _active = policy


class use_vp_policy:
    """Context manager: activate `policy` for a scope, restore the prior
    one. For per-parse activation (Driftmoor) and hermetic tests."""

    def __init__(self, policy: VPGroundingPolicy):
        self._new = policy
        self._prev: Optional[VPGroundingPolicy] = None

    def __enter__(self) -> VPGroundingPolicy:
        global _active
        self._prev = _active
        _active = self._new
        return self._new

    def __exit__(self, *exc) -> None:
        global _active
        _active = self._prev


__all__ = [
    "VPGroundingPolicy",
    "EngrafVPPolicy",
    "PermissiveVPPolicy",
    "get_active_vp_policy",
    "set_active_vp_policy",
    "use_vp_policy",
]
