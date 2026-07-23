"""Parser-honesty diagnostics: which words the active lexicon does not know.

Host-neutral. When the parser cannot fully tokenize a line, this reports the
unrecognized words -- typos, out-of-vocabulary terms -- and, where a known word
is close, suggests it. It lets a host tell the user *which words* it did not
understand (and what it may have meant) instead of silently handing the line to
a language model.

All functions read the currently active lexicon, so a host must have installed
its vocabulary (``set_active_lexicon`` / ``use_lexicon``) before calling.
"""

from typing import List, Tuple

from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_best
from latn.lexer.lexicon import get_active_lexicon


def _damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string-alignment distance: insertions, deletions, substitutions,
    and *adjacent transpositions* each cost 1. Transposition support matters --
    "teh"->"the" is one of the commonest typos and plain Levenshtein scores it 2.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def unknown_words(text: str) -> List[str]:
    """The words in ``text`` the active lexicon does not recognize.

    Order-preserving and de-duplicated (case-insensitively). Uses the
    tokenizer's own judgement, so morphology and compounds are respected: a
    plural whose singular is known, an inflected verb/adjective, or a known
    multi-word phrase are NOT reported -- only genuinely unrecognized tokens
    (which the tokenizer marks with the ``unknown`` dimension).
    """
    seen: set = set()
    out: List[str] = []
    for tok in latn_tokenize_best(text or ""):
        if getattr(tok, "isa", None) and tok.isa("unknown"):
            word = (getattr(tok, "word", "") or "").strip()
            # Only report genuine words -- skip stray punctuation/symbol tokens
            # (a bare quote from quoted speech, "@", ...) which aren't vocabulary
            # and would produce nonsense suggestions.
            if not any(ch.isalnum() for ch in word):
                continue
            key = word.lower()
            if key not in seen:
                seen.add(key)
                out.append(word)
    return out


def _candidate_vocabulary() -> List[str]:
    """Single-word alphabetic keys of the active lexicon -- the pool corrections
    are drawn from. Compounds and punctuation make poor spelling hints."""
    return [w for w in get_active_lexicon()
            if isinstance(w, str) and " " not in w and w.isalpha()]


def suggest_words(word: str, *, n: int = 3, max_distance: int = None) -> List[str]:
    """Known words within a small edit distance of ``word`` (a likely intended
    spelling), closest first. Empty when nothing is near -- so a genuine unknown
    word (not a typo) gets no misleading guess.

    ``max_distance`` defaults to a length-aware bound: 1 for short words (<= 5),
    2 for longer ones, so "teh"->"the" and "crosbow"->"crossbow" both resolve
    while "floober" matches nothing.
    """
    w = (word or "").lower()
    if not w or not any(ch.isalnum() for ch in w):
        return []
    bound = max_distance if max_distance is not None else (2 if len(w) >= 6 else 1)
    scored: List[Tuple[int, str]] = []
    for cand in _candidate_vocabulary():
        if abs(len(cand) - len(w)) > bound:   # distance >= length difference
            continue
        dist = _damerau_levenshtein(w, cand)
        if dist <= bound:
            scored.append((dist, cand))
    scored.sort(key=lambda ds: (ds[0], ds[1]))
    return [cand for _, cand in scored[:n]]


def diagnose(text: str, *, n: int = 3, max_distance: int = None) -> List[Tuple[str, List[str]]]:
    """Pair each unknown word in ``text`` with its nearest known words (possibly
    empty). The raw material for a host's "I don't know 'X' -- did you mean 'Y'?"
    reply."""
    return [(w, suggest_words(w, n=n, max_distance=max_distance))
            for w in unknown_words(text)]
