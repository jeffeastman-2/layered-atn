import re
from latn.lexer.lexicon import get_active_lexicon

# Common verb inflection patterns
VERB_INFLECTION_PATTERNS = [
    # Past participle patterns (for -ed ending)
    (r"(.+)ed$", "verb_past_part", lambda m: [m.group(1), m.group(1) + "e"]),  # called -> call, named -> name
    
    # Present participle / gerund patterns (for -ing ending) 
    # Sets verb_present_part; caller should also set gerund if used nominally
    (r"(.+)ing$", "verb_present_part", lambda m: [m.group(1), m.group(1) + "e"]),  # calling -> call, naming -> name
]

# Irregular verb forms that don't follow standard patterns
# Note: Only include forms that are NOT already in the main vocabulary
IRREGULAR_VERB_FORMS = {
    # Common irregular past participles (only if not in main vocab)
    "done": ("do", "verb_past_part"), 
    "gone": ("go", "verb_past_part"),
    "seen": ("see", "verb_past_part"),
    "taken": ("take", "verb_past_part"),
    "given": ("give", "verb_past_part"),
    "made": ("make", "verb_past_part"),
    "said": ("say", "verb_past_part"),
    "told": ("tell", "verb_past_part"),
    "found": ("find", "verb_past_part"),
    
    # Common irregular past tense forms (only if not in main vocab)
    "had": ("have", "verb_past"),
    "did": ("do", "verb_past"),
    "went": ("go", "verb_past"),
    "saw": ("see", "verb_past"),
    "took": ("take", "verb_past"),
    "gave": ("give", "verb_past"),
    "said": ("say", "verb_past"),
    "told": ("tell", "verb_past"),
    "found": ("find", "verb_past"),
}

def find_root_verb(inflected_word):
    """
    Try to find the root form of an inflected verb.
    Returns (root_verb, inflection_type, found) tuple where:
    - root_verb is the base form 
    - inflection_type is the verb inflection dimension to set
    - found is True if a root was found in vocabulary
    """
    word_lower = inflected_word.lower()
    
    # Check if it's already a known word
    if word_lower in get_active_lexicon():
        return word_lower, None, True
    
    # Check irregular forms first
    if word_lower in IRREGULAR_VERB_FORMS:
        root, inflection_type = IRREGULAR_VERB_FORMS[word_lower]
        if root in get_active_lexicon():
            # Check if root has verb or tobe dimension (both are verb-like)
            root_vector = get_active_lexicon()[root]
            if root_vector["verb"] > 0 or root_vector["tobe"] > 0:
                return root, inflection_type, True
    
    # Try regular inflection patterns
    for pattern, inflection_type, root_candidates_func in VERB_INFLECTION_PATTERNS:
        match = re.match(pattern, word_lower)
        if match:
            # Try each possible root form
            candidates = root_candidates_func(match)
            for candidate in candidates:
                if candidate in get_active_lexicon():
                    # Check if candidate has verb or tobe dimension
                    candidate_vector = get_active_lexicon()[candidate]
                    if candidate_vector["verb"] > 0 or candidate_vector["tobe"] > 0:
                        return candidate, inflection_type, True
    
    return word_lower, None, False

def is_verb_inflection(word):
    """
    Check if a word appears to be a verb inflection that could have a root form.
    """
    _, _, found = find_root_verb(word)
    return found and word.lower() not in get_active_lexicon()


def verb_to_gerund(verb: str) -> str:
    """
    Convert a base verb to its gerund (-ing) form.
    
    Handles common English spelling rules:
    - drop silent 'e': make -> making, create -> creating
    - double final consonant after short vowel: run -> running, put -> putting
    - verbs ending in 'ie': die -> dying, lie -> lying
    - regular: draw -> drawing, move -> moving
    
    Args:
        verb: Base form of verb (e.g., 'draw', 'make', 'run')
        
    Returns:
        Gerund form (e.g., 'drawing', 'making', 'running')
    """
    verb = verb.lower().strip()
    
    if not verb:
        return verb
    
    # Irregular/exception cases
    irregular_gerunds = {
        'begin': 'beginning',
        'forget': 'forgetting', 
        'occur': 'occurring',
        'refer': 'referring',
        'prefer': 'preferring',
        'open': 'opening',  # Don't double 'n' despite CVC pattern
    }
    if verb in irregular_gerunds:
        return irregular_gerunds[verb]
    
    # Verbs ending in 'ie' -> 'ying' (die -> dying, lie -> lying)
    if verb.endswith('ie'):
        return verb[:-2] + 'ying'
    
    # Verbs ending in 'ee' -> just add 'ing' (see -> seeing, free -> freeing)
    if verb.endswith('ee'):
        return verb + 'ing'
    
    # Verbs ending in silent 'e' -> drop 'e' and add 'ing'
    # (make -> making, create -> creating, move -> moving)
    if verb.endswith('e') and len(verb) > 1:
        return verb[:-1] + 'ing'
    
    # Double final consonant for CVC pattern (consonant-vowel-consonant)
    # Only for short one-syllable words ending in single consonant after single vowel
    # Common cases: run -> running, put -> putting, sit -> sitting
    vowels = 'aeiou'
    if len(verb) >= 3:
        last = verb[-1]
        second_last = verb[-2]
        third_last = verb[-3]
        
        # Don't double 'w', 'x', 'y'
        # Only double if it's a short (typically one-syllable) word
        # Multi-syllable words like 'color', 'position', 'open' don't double
        if last not in vowels and last not in 'wxy':
            if second_last in vowels and third_last not in vowels:
                # Only double for short words (3-4 chars typically)
                # This avoids doubling in longer words like 'color', 'enter', 'open'
                if len(verb) <= 4:
                    return verb + last + 'ing'
    
    # Regular case: just add 'ing'
    return verb + 'ing'
