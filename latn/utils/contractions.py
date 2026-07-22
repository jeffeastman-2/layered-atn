"""Contraction expansion for the LATN tokenizer.

Language-general, so it lives in the core beside the noun and verb inflectors.
The raw tokenizer regex splits a contraction on its apostrophe -- ``I'd`` comes
in as ``["I", "'", "d"]`` -- and this turns those three tokens back into words.

Unambiguous clitics expand to a single reading. Ambiguous ones expand to
several, returned as separate token lists so the tokenizer can emit each as its
own hypothesis and let the grammar keep whichever parses:

- ``'d``  -> "would" | "had"
- ``'s``  -> "is" | "has" | possessive (drop the clitic, e.g. "Jorgen's ledger")
"""

from typing import List, Optional

# tail -> single expansion word.
_UNAMBIGUOUS = {"m": "am", "re": "are", "ll": "will", "ve": "have"}

# tail -> list of expansions; [] means "drop the clitic" (the possessive reading
# of 's, which lets a bare noun + noun parse: "shopkeeper's ledger").
_AMBIGUOUS = {
    "d": [["would"], ["had"]],
    "s": [["is"], ["has"], []],
}

# n't stems whose base is not just "X minus a trailing n".
_NT_IRREGULAR = {"can": ["can"], "won": ["will"], "shan": ["shall"], "ain": ["is"]}

_TAILS = set(_UNAMBIGUOUS) | set(_AMBIGUOUS) | {"t"}


def _nt_expansion(x: str) -> Optional[List[str]]:
    """Expand an n't contraction from the word before "'t" (``don``, ``won``,
    ``isn`` ...). Returns e.g. ``["do", "not"]`` or None if it isn't an n't."""
    xl = x.lower()
    if xl in _NT_IRREGULAR:
        return _NT_IRREGULAR[xl] + ["not"]
    if xl.endswith("n"):
        return [x[:-1], "not"]
    return None


def _variants_for(x: str, tail: str) -> Optional[List[List[str]]]:
    """The expansion variant(s) for ``x`` + "'" + ``tail`` (each a token list),
    or None when this isn't a contraction we expand."""
    if tail == "t":
        nt = _nt_expansion(x)
        return [nt] if nt else None
    if tail in _UNAMBIGUOUS:
        return [[x, _UNAMBIGUOUS[tail]]]
    if tail in _AMBIGUOUS:
        return [[x] + exp for exp in _AMBIGUOUS[tail]]
    return None


def expand_contractions(raw_tokens: List[str], *, max_variants: int = 32) -> List[List[str]]:
    """Expand contractions in the tokenizer's raw token list.

    Returns one or more token lists -- several when an ambiguous clitic branches
    (the cartesian product over branch points). Non-contraction input returns a
    single unchanged list, so callers with no contractions are unaffected.
    ``max_variants`` caps the product against pathological stacking.
    """
    segments: List[List[List[str]]] = []
    i, n = 0, len(raw_tokens)
    while i < n:
        if i + 2 < n and raw_tokens[i + 1] == "'" and raw_tokens[i + 2].lower() in _TAILS:
            variants = _variants_for(raw_tokens[i], raw_tokens[i + 2].lower())
            if variants is not None:
                segments.append(variants)
                i += 3
                continue
        segments.append([[raw_tokens[i]]])
        i += 1

    results: List[List[str]] = [[]]
    for seg in segments:
        results = [r + v for r in results for v in seg]
        if len(results) > max_variants:
            results = results[:max_variants]
    return results
